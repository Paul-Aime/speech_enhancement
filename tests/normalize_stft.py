import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    c, h, w = 1, 3, 7
    
    S = torch.randint(-7, 21, (c, h, w)).to(dtype=torch.double)

    Sn_std = normalize_stft(S, mode='std', inplace=False)
    Sn_max = normalize_stft(S, mode='max', inplace=False)
    
    # --- Display
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    
    im1 = ax1.imshow(S.squeeze())
    ax1.set_title('S')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = ax2.imshow(Sn_std.squeeze())
    ax2.set_title('Sn_std')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    
    
    im3 = ax3.imshow(Sn_max.squeeze())
    ax3.set_title('Sn_max')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax, orientation='vertical')
    
    plt.show()


def normalize_stft(S, mode='std', inplace=True):
    # Only works for 1 channel
    # shape of S : (1, H, W)
    # see torchvision.transforms.functional import normalize for C>1
    
    if not inplace:
        S = S.clone()

    if mode == 'max':
        return S.sub_(S.mean()).div_(S.abs().max())
    elif mode == 'std':
        return S.sub_(S.mean()).div_(S.std())
    else:
        print('ERROR : mode unknown')
        return


if __name__ == "__main__":
    main()
