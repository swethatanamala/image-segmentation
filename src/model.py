import torch
from torch import nn

class _EncoderBlock(nn.Module):
    """Conv block of U-Net"""

    def __init__(self, fin, fout):
        super(_EncoderBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.block = nn.Sequential(
            self._conv_relu(fin, fout),
            self._conv_relu(fout, fout)
        )

    def _conv_relu(self, fin, fout):
        layers = [
            nn.Conv2d(fin, fout, 3, stride=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Deconv block of U-Net"""

    def __init__(self, fin, fout):
        super(_DecoderBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(fin, fout, kernel_size=2, stride=2)
        )
        self.block = nn.Sequential(
            self._conv_relu(fin, fout),
            self._conv_relu(fout, fout)
        )

    def _conv_relu(self, fin, fout):
        layers = [
            nn.Conv2d(fin, fout, 3, stride=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def _crop_and_copy(self, deconv_out, y):
        print(y.size())
        y_x = y.size()[2]
        deconv_out_x = deconv_out.size()[2]
        y_y = y.size()[3]
        deconv_out_y = deconv_out.size()[3]      
        start_ind_x = y_x//2 - deconv_out_x//2
        end_ind_x = y_x//2 + (deconv_out_x - deconv_out_x//2)
        start_ind_y = y_y//2 - deconv_out_y//2
        end_ind_y = y_y//2 + (deconv_out_y - deconv_out_y//2)
        concat_layer = torch.cat([y[:, :, start_ind_x:end_ind_x, start_ind_y:end_ind_y], deconv_out], 1)
        return concat_layer

    def forward(self, x, y):
        deconv_out = self.upsample(x)
        print(deconv_out.size())
        print(y.size())
        concat_layer = self._crop_and_copy(deconv_out, y)
        return self.block(concat_layer)

class UNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.config = [((num_channels, 64), (128, 64)),
                                ((64, 128), (256, 128)),
                               ((128, 256), (512, 256)),
                               ((256, 512), (1024, 512)),
                               (     (512, 1024)       )
                      ]
        self.depth = len(self.config)
        self._create_blocks()

    def _create_blocks(self):

        for i, v in enumerate(self.config[:-1]):
            enc_block, dec_block = v[0], v[1]

            self.add_module('encoder_{}'.format(i), _EncoderBlock(*enc_block))
            self.add_module('decoder_{}'.format(i), _DecoderBlock(*dec_block))

        # base layer
        base = _EncoderBlock(*self.config[-1])
        self.add_module('encoder_{}'.format(self.depth - 1), base)

        # final output layer
        final_feature_maps = self.config[0][1][1]
        output = nn.Conv2d(final_feature_maps, self.num_classes, 1, stride=1)
        self.add_module('output', output)

        # pooling layer (downsampling layer)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        encoded = [self.encoder_0(x)]
        for i in range(1, self.depth):
            max_pooled = self.pool(encoded[i - 1])
            encoder_block = getattr(self, 'encoder_{}'.format(i))
            encoded.append(encoder_block(max_pooled))

        #upsampling path
        out = encoded[self.depth - 1]
        print(out.size())
        for i in range(self.depth - 2, -1, -1):
            decoder_block = getattr(self, 'decoder_{}'.format(i))
            out = decoder_block(out, encoded[i])

        #final prediction
        out = self.output(out)
        return out

