import numpy as np
import cv2
from numbers import Number

class NDTransform(object):
    """Base class for all numpy based transforms.

    This class achieves the following:

    * Abstract the transform into
        * Getting parameters to apply which is only run only once per __call__.
        * Applying transform given parameters
    * Check arguments passed to a transforms for consistency

    Abstraction is especially useful when there is randomness involved with the
    transform. You don't want to have different transforms applied to different
    members of a data point.
    """

    def _argcheck(self, data):
        """Check data for arguments."""
        
        if isinstance(data, np.ndarray):
            assert data.ndim in {2, 3}, \
                'Image should be a ndarray of shape H x W x C or H X W.'
            if data.ndim == 3:
                assert data.shape[2] < data.shape[0], \
                    'Is your color axis the last? Roll axes using np.rollaxis.'

            return data.shape[:2]
        elif isinstance(data, dict):
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    assert isinstance(k, str)

            shapes = {k: self._argcheck(img) for k, img in data.items()
                      if isinstance(img, np.ndarray)}
            assert len(set(shapes.values())) == 1, \
                'All member images must have same size. Instead got: {}'.format(shapes)
            return set(shapes.values()).pop()
        else:
            raise TypeError('ndarray or dict of ndarray can only be passed')

    def _get_params(self, h, w, seed=None):
        """Get parameters of the transform to be applied for all member images.

        Implement this function if there are parameters to your transform which
        depend on the image size. Need not implement it if there are no such
        parameters.

        Parameters
        ----------
        h: int
            Height of the image. i.e, img.shape[0].
        w: int
            Width of the image. i.e, img.shape[1].

        Returns
        -------
        params: dict
            Parameters of the transform in a dict with string keys.
            e.g. {'angle': 30}
        """
        return {}

    def _transform(self, img, is_label, **kwargs):
        """Apply the transform on an image.

        Use the parameters returned by _get_params and apply the transform on
        img. Be wary if the image is label or not.

        Parameters
        ----------
        img: ndarray
            Image to be transformed. Can be a color (H X W X C) or
            gray (H X W)image.
        is_label: bool
            True if image is to be considered as label, else False.
        **kwargs
            kwargs will be the dict returned by get_params

        Return
        ------
        img_transformed: ndarray
            Transformed image.
        """
        raise NotImplementedError

    def __call__(self, data, seed=None):
        """
        Parameters
        ----------
        data: dict or ndarray
            Image ndarray or a dict of images. All ndarrays in the dict are
            considered as images and should be of same size. If key for a
            image in dict has string `target` in it somewhere, it is
            considered as a target segmentation map.
        """
        h, w = self._argcheck(data)
        params = self._get_params(h, w, seed=seed)

        if isinstance(data, np.ndarray):
            return self._transform(data, is_label=False, **params)
        else:
            data = data.copy()
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    if isinstance(k, str) and 'target' in k:
                        is_label = True
                    else:
                        is_label = False

                    data[k] = self._transform(img.copy(), is_label, **params)
            return data
        
class Compose(NDTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, seed=None):
        for t in self.transforms:
            data = t(data, seed)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class RandomHorizontalFlip(NDTransform):
    """Flip horizontally with 0.5 probability."""

    def _get_params(self, h, w, seed=None):
        rng = np.random.RandomState(seed)
        return {'to_flip': rng.uniform() < 0.5}

    def _transform(self, img, is_label, to_flip):
        if to_flip:
            return np.flip(img, 1)
        else:
            return img
        
class RandomIntensityJitter(NDTransform):
    """Random jitters in brightness, contrast and saturation.

    Parameters
    ----------
    brightness: float
        Intensity of brightness jitter. float in [0, 1].
    contrast: float
        Intensity of contrast jitter. float in [0, 1].
    saturation: float
        Intensity of saturation jitter. float in [0, 1].
    gamma: float
        Intensity of gamma jitter. float in [0, 1].
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, gamma=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma

    def _blend(self, img1, img2, alpha):
        return img1 * alpha + img2 * (1 - alpha)

    def _gs(self, img):
        if img.ndim == 3:
            return rgb2gray(img)[:, :, np.newaxis]
        else:
            return img

    def _saturation(self, img, alpha):
        if alpha == 1:
            return img

        return self._blend(img, self._gs(img), alpha)

    def _brightness(self, img, alpha):
        if alpha == 1:
            return img

        return self._blend(img, np.zeros_like(img), alpha)

    def _contrast(self, img, alpha):
        if alpha == 1:
            return img

        mean_img = np.full_like(img, self._gs(img).mean())
        return self._blend(img, mean_img, alpha)

    def _gamma(self, img, alpha):
        if alpha == 1:
            return img

        gain = 1
        return gain * (img**alpha)

    def _get_params(self, h, w, seed=None):
        rng = np.random.RandomState(seed)
        vars = [1 + self.brightness * rng.uniform(-1, 1),
                1 + self.contrast * rng.uniform(-1, 1),
                1 + self.saturation * rng.uniform(-1, 1),
                1 + self.gamma * rng.uniform(-1, 1)]

        return {'vars': vars, 'order': rng.permutation(len(vars))}

    def _transform(self, img, is_label, vars, order):
        if is_label:
            return img
        else:
            tsfrms = [self._brightness, self._contrast, self._saturation,
                      self._gamma]
            for i in order:
                img = tsfrms[i](img, vars[i])

            return img
        
class Clip(NDTransform):
    """Clip numpy array by a low and high value.

    Linear transformation can be applied after clipping so that new desired
    high and lows can be set.

    Default behaviour is to clip with 0 and 1 as bounds.

    Parameters
    ----------
    inp_low: float, optional
        Minimum value of the input, default is 0.
    inp_high: float, optional
        Maximum value of the input, default is 1,
    out_low: float, optional
        New minimum value for the output, default is inp_low.
    out_high: float, optional
        New Maximum value for the output, default is inp_high.

    """

    def __init__(self, inp_low=0, inp_high=1, out_low=None, out_high=None):
        self.inp_low = inp_low
        self.inp_high = inp_high
        self.out_low = out_low if out_low is not None else inp_low
        self.out_high = out_high if out_high is not None else inp_high

    def _transform(self, img, is_label):
        if is_label:
            return img
        else:
            img = (img - self.inp_low) / (self.inp_high - self.inp_low)
            img = np.clip(img, 0, 1)
            img = self.out_low + (self.out_high - self.out_low) * img

            return img
        
class ToTensor(NDTransform):
    """Convert ndarrays to tensors.

    Following are taken care of when converting to tensors:

    * Axes are swapped so that color axis is in front of rows and columns
    * A color axis is added in case of gray images
    * Target images are left alone and are directly converted
    * Label images is set to LongTensor by default as expected by torch's loss
      functions.

    Parameters
    ----------
    dtype: torch dtype
        If you want to convert all tensors to cuda, you can directly
        set dtype=torch.cuda.FloatTensor. This is for non label images
    dtype_label: torch dtype
        Same as above but for label images.
    """


    def _transform(self, img, is_label):
        import torch
        img = np.ascontiguousarray(img)
        if not is_label:
            # put it from HWC to CHW format
            if img.ndim == 3:
                img = np.rollaxis(img, 2, 0)
            elif img.ndim == 2:
                img = img.reshape((1,) + img.shape)
        else:
            if img.ndim == 3:  # making transforms work for multi mask models
                img = np.rollaxis(img, 2, 0)

        img = torch.from_numpy(img)

        if is_label:
            return img.long()
        else:
            return img.float()

class _RandomSizedCrop(NDTransform):
    """Randomly sized crop within a specified size and aspect ratio range.

    An area fraction and a aspect ratio is sampled within frac_range and
    aspect_range respectively. Then sides of the crop are calculated using
    these two. Random crop of this size is finally resized to desired size.

    Parameters
    ----------
    output_shape: tuple
        `(rows, cols)` of output image.
    frac_range: sequence of length 2
        Range for fraction of the area to be sampled from.
    aspect_range: sequence of length 2
        Aspect ratio range to be sampled from.
    **kwargs: optional
        Other kwargs as described in skimage.transforms.resize
        or lycon.resize
    """

    def __init__(self, output_shape, frac_range=[0.08, 1],
                 aspect_range=[3 / 4, 4 / 3], **kwargs):
        if isinstance(output_shape, Number):
            self.output_shape = (output_shape, output_shape)
        else:
            assert len(output_shape) == 2
            self.output_shape = output_shape

        self.frac_range = frac_range
        self.log_aspect_range = np.log(aspect_range)
        self.kwargs = kwargs

    def _get_params(self, h, w, seed=None):
        rng = np.random.RandomState(seed)
        area = h * w

        attempts = 0
        while attempts < 10:
            try:
                targer_area = area * rng.uniform(*self.frac_range)
                aspect_ratio = np.exp(
                    rng.uniform(*self.log_aspect_range))

                new_h = int(np.sqrt(targer_area * aspect_ratio))
                new_w = int(np.sqrt(targer_area / aspect_ratio))

                if rng.uniform() < 0.5:
                    new_h, new_w = new_w, new_h

                assert (new_h <= h) and (new_w <= w), 'Attempt failed'

                h1 = 0 if h == new_h else rng.randint(0, h - new_h)
                w1 = 0 if w == new_w else rng.randint(0, w - new_w)

                return {'h1': h1, 'w1': w1, 'new_h': new_h, 'new_w': new_w}
            except AssertionError:
                attempts += 1

        # fall back
        new_size = min(h, w)
        h1, w1 = (h - new_size) // 2, (w - new_size) // 2

        return {'h1': h1, 'w1': w1, 'new_h': new_size, 'new_w': new_size}

class _Resize(NDTransform):
    """Resize image to match a certain size.

    Parameters
    ----------
    output_shape : int or tuple
        Size of the generated output image `(rows, cols)`. If it is a
        number, aspect ratio of the image is preserved and smaller of the
        height and width is matched to it.
    **kwargs: optional
        Other params as described in skimage.transforms.resize
        or lycon.resize
    """

    def __init__(self, output_shape, **kwargs):
        self.kwargs = kwargs
        self.output_shape = output_shape

    def _get_params(self, h, w, seed=None):
        if isinstance(self.output_shape, Number):
            req = self.output_shape
            if h > w:
                output_shape = (int(h * req / w), req)
            else:
                output_shape = (req, int(w * req / h))
        else:
            output_shape = self.output_shape

        return {'output_shape': output_shape}

class Resize(_Resize):
    __doc__ = _Resize.__doc__

    def _transform(self, img, is_label, output_shape):
        return _cv2_resize(img, is_label, output_shape, self.kwargs)

def _cv2_resize(img, is_label, output_shape, kwargs):
    kwargs = kwargs.copy()
    if is_label:
        kwargs.update({'interpolation': cv2.INTER_NEAREST})

    return cv2.resize(img, (output_shape[1], output_shape[0]), **kwargs)

def rgb2gray(rgb):
    if rgb.ndim == 3:
        return (0.2125 * rgb[..., 0] +
                0.7154 * rgb[..., 1] +
                0.0721 * rgb[..., 2])
    elif rgb.ndim == 2:
        return rgb
    else:
        raise ValueError('Invalid Dimensions')

class RandomSizedCrop(_RandomSizedCrop):
    __doc__ = _RandomSizedCrop.__doc__

    def _transform(self, img, is_label, h1, w1, new_h, new_w):
        img = img[h1: h1 + new_h, w1: w1 + new_w]
        return _cv2_resize(img, is_label, self.output_shape, self.kwargs)
    
class _RandomRotate(NDTransform):
    """Randomly rotate image.

    Parameters
    ----------
    angle_range: float or tuple
        Range of angles in degrees. If float, angle_range = (-theta, theta).
    kwargs: optional
        Other kwargs as described in skimage.transforms.rotate
    """

    def __init__(self, angle_range, **kwargs):
        """Angle is in degrees."""
        if isinstance(angle_range, Number):
            assert angle_range > 0
            self.angle_range = (-angle_range, angle_range)
        else:
            self.angle_range = angle_range

        self.kwargs = kwargs

    def _get_params(self, h, w, seed=None):
        rng = np.random.RandomState(seed)
        angle = rng.uniform(*self.angle_range)
        return {'angle': angle}

    
class RandomRotate(_RandomRotate):
    __doc__ = _RandomSizedCrop.__doc__

    def _transform(self, img, is_label, angle):
        return _cv2_rotate(img, is_label, angle, self.kwargs)
    
def _cv2_rotate(img, is_label, theta, kwargs):
    if is_label:
        kwargs.update({'flags': cv2.INTER_NEAREST})

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    img = cv2.warpAffine(img, M, (cols, rows), **kwargs)
    return img