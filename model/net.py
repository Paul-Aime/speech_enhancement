import torch
import torch.nn as nn

# from utils.cuda_utils import init_cuda

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


class MyCNN(torch.nn.Module):
    """
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    """

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self, params):

        super(MyCNN, self).__init__()

        # TODO add self.conv_layer that contains batch norm + relu
        # see https://github.com/zhr1201/CNN-for-single-channel-speech-enhancement/blob/master/SENN.py

        # Architecture
        # To understand what will be the output size, see :
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        self.enc_conv1 = self.ConvLayer(1, 12, kernel_size=(13, params.n_frames))
        self.enc_conv2 = self.ConvLayer(12, 16, kernel_size=(11, 1))
        self.enc_conv3 = self.ConvLayer(16, 20, kernel_size=(9, 1))
        self.enc_conv4 = self.ConvLayer(20, 24, kernel_size=(7, 1))

        self.latent = self.ConvLayer(24, 32, kernel_size=(7, 1))

        self.dec_conv1 = self.ConvLayer(32, 24, kernel_size=(7, 1))
        self.dec_conv2 = self.ConvLayer(24, 20, kernel_size=(9, 1))
        self.dec_conv3 = self.ConvLayer(20, 16, kernel_size=(11, 1))
        self.dec_conv4 = self.ConvLayer(16, 12, kernel_size=(13, 1))

        self.out_conv = self.ConvLayer(12, 1, kernel_size=(params.n_fft//2 + 1, 1))

        self.double()
        self.to(DEVICE)

    def forward(self, x):

        # shape of x : (B, c, H, W)

        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)

        x = self.latent(x)

        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        x = self.dec_conv4(x)

        x = self.out_conv(x)

        return x

    def ConvLayer(self, in_channels, out_channels, kernel_size):

        # For it to be as in park2017, `kernel_size[1]` must be equal to
        # the incoming input width. Hence nframes on the first layer, then 1.

        assert kernel_size[0] % 2
        pad_h = (kernel_size[0] - 1) // 2

        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=(pad_h, 0)),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

        return layer
