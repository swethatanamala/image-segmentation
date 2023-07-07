"""U-Net Model."""
import torch
from torch import nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


class _EncoderBlock(nn.Module):
    """Conv block of U-Net."""

    def __init__(self, fin, fout, dropout=0):
        super(_EncoderBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.block = nn.Sequential(
            self._conv_bn_relu(fin, fout, dropout),
            self._conv_bn_relu(fout, fout, dropout)
        )

    def _conv_bn_relu(self, fin, fout, dropout=0):
        layers = [
            nn.Conv2d(fin, fout, 3, stride=1, padding=1),
            nn.BatchNorm2d(fout),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def __repr__(self):
        return "Encoder Block ({} -> {})".format(self.fin, self.fout)


class _DecoderBlock(nn.Module):
    """Deconv block of U-Net."""

    def __init__(self, fin, fout):
        super(_DecoderBlock, self).__init__()
        assert fin == 2 * fout, 'channels mismatch'
        self.fin = fin
        self.fout = fout
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(fin, fout, kernel_size=2, stride=2),
            nn.BatchNorm2d(fout)
        )
        self.conv = _EncoderBlock(fin, fout)

    def forward(self, x, y):
        deconv_out = self.upsample(x)
        skip_layer_out = torch.cat([y, deconv_out], 1)
        return self.conv(skip_layer_out)

    def __repr__(self):
        return "Decoder Block ({} -> {})".format(self.fin, self.fout)


class UNet(nn.Module):
    """Unet"""

    def __init__(self, num_channels=1, num_classes=2, config=None):
        """
        Parameters
        ----------
        config: list
            list of tuples of tuples.
        """
        super(UNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        if config is None:
            self.config = [((num_channels, 64), (128, 64)),
                           ((64, 128), (256, 128)),
                           ((128, 256), (512, 256)),
                           ((256, 512), (1024, 512)),
                           (    (512, 1024, 0.5)   )]
        else:
            self.config = config

        self.depth = len(self.config)
        self._create_blocks(self.config, num_classes)

    def _create_blocks(self, config, num_classes):
        logger.info('Building Unet')
        depth = len(config)

        # upsampling and downsampling layers
        for i, v in enumerate(config[:-1]):
            enc_block, dec_block = v[0], v[1]

            self.add_module('encoder_{}'.format(i), _EncoderBlock(*enc_block))
            self.add_module('decoder_{}'.format(i), _DecoderBlock(*dec_block))

        # base layer
        base = _EncoderBlock(*config[-1])
        self.add_module('encoder_{}'.format(depth - 1), base)

        # final output layer
        final_feature_maps = config[0][1][1]
        output = nn.Conv2d(final_feature_maps, num_classes, 3,
                           stride=1, padding=1)
        self.add_module('output', output)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward_encoder(self, x):
        # downsampling path
        encoded = [self.encoder_0(x)]
        for i in range(1, self.depth):
            max_pooled = self.pool(encoded[i - 1])
            encoder_block = getattr(self, 'encoder_{}'.format(i))
            encoded.append(encoder_block(max_pooled))

        return encoded

    def forward_decoder(self, encoded):
        # upsampling path
        out = encoded[self.depth - 1]
        for i in range(self.depth - 2, -1, -1):
            decoder_block = getattr(self, 'decoder_{}'.format(i))
            out = decoder_block(out, encoded[i])

        return out

    def forward(self, x):
        encoded = self.forward_encoder(x)
        out = self.forward_decoder(encoded)
        out = self.output(out)
        return out


if __name__ == '__main__':
    model = UNet(4, 3)
    test = torch.rand(2, 4, 512, 512)
    out = model(test)
    print(model)
    print('input: {}'.format(test.size()))
    print('output: {}'.format(out.size()))
