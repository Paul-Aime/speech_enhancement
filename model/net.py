import torch
from torch.nn import Conv2d

from utils.cuda_utils import init_cuda

DEVICE = init_cuda()


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
        self.enc_conv1 = Conv2d(1, 12, kernel_size=(13, params.n_frames),
                                padding=(6, 0))
        self.enc_conv2 = Conv2d(12, 16, kernel_size=(11, 1), padding=(5, 0))
        self.enc_conv3 = Conv2d(16, 20, kernel_size=(9, 1), padding=(4, 0))
        self.enc_conv4 = Conv2d(20, 24, kernel_size=(7, 1), padding=(3, 0))

        self.latent = Conv2d(24, 32, kernel_size=(7, 1), padding=(3, 0))

        self.dec_conv1 = Conv2d(32, 24, kernel_size=(7, 1), padding=(3, 0))
        self.dec_conv2 = Conv2d(24, 20, kernel_size=(9, 1), padding=(4, 0))
        self.dec_conv3 = Conv2d(20, 16, kernel_size=(11, 1), padding=(5, 0))
        self.dec_conv4 = Conv2d(16, 12, kernel_size=(13, 1), padding=(6, 0))

        self.out_conv = Conv2d(12, 1, kernel_size=(params.n_fft//2 + 1, 1),
                               padding=(((params.n_fft//2 + 1)-1)//2, 0))

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
