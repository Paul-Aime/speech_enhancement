import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import librosa.display
import librosa
import numpy as np


def main():

    # Load image
    im = plt.imread('./tests/stft_im.png')[:, :, 1]
    im = torch.as_tensor(im).unsqueeze(0)

    # Parameters to get from params object
    n_fft = 256
    fs = 8000

    # --- Figure
    fig, ax2 = plt.subplots(nrows=1, ncols=1)

    # Original image
    # im1 = ax1.imshow(im.squeeze(), origin='lower')
    # ax1.set_xlabel('STFT frame number')
    # ax1.set_ylabel('Frequencies')
    # ax1.set_title('Original')

    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')

    # Custom image
    # im2 = ax2.imshow(im.squeeze(), origin='lower', cmap=plt.get_cmap('magma'), vmin=0, vmax=1)
    im2 = ax2.pcolor(im.squeeze(), cmap=plt.get_cmap('magma'), vmin=0, vmax=1)
    ax2.set_xlabel('STFT frame number')
    ax2.set_ylabel('Frequencies')
    ax2.set_title('Custom')

    yticks_step_in_hz = 500
    yticks_hz = np.arange(0, fs/2+1, yticks_step_in_hz)
    yticks = yticks_hz * (n_fft/fs)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks_hz.astype(np.int))

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig('./tests/stft_plot.png')
    plt.show()


if __name__ == '__main__':
    main()
