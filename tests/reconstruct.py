import torch
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import stft as librosa_stft
from librosa.core import istft as librosa_istft

if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    # https://stackoverflow.com/a/27876800/10076676
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from model.dataset import stft, istft, normalize_sound, normalize_stft


def main():
    signame = './tests/data/SA1_noisy.signal'
    x = torch.load(signame)

    stft_kwargs = {
        "n_fft": 256,
        "hop_length": 256 // 2,
        "win_length": None,
        "window": torch.hann_window(256).numpy(),
        "center": True,
        "dtype": np.complex64,
        "pad_mode": 'reflect'
    }

    istft_kwargs = {
        "hop_length": stft_kwargs['hop_length'],
        "win_length": stft_kwargs['win_length'],
        "window": stft_kwargs['window'],
        "center": stft_kwargs['center']
    }

    # --- The way used in the code
    S_abs, S_ang = stft(x, **stft_kwargs)
    
    # S_abs = normalize_stft(S_abs)
    # S_ang = normalize_stft(S_ang)
    
    y = istft(S_abs, S_ang, length=None, **istft_kwargs)
    y = torch.as_tensor(y, dtype=torch.double).unsqueeze(0)
    # y = dataset.normalize_sound(y, inplace=True)
    
    # --- Using directly librosa
    y_librosa = librosa_istft(librosa_stft(x.squeeze().numpy(), **stft_kwargs), **istft_kwargs)
    
    # --- Figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    
    ax1.plot(x.squeeze())
    ax1.set_title('Original signal')
    
    ax2.plot(y.squeeze())
    ax2.set_title('Retrieved from code method')
    
    ax3.plot(y_librosa)
    ax3.set_title('Retrieved from starightforward istft(stft(.)), only librosa')
    
    plt.show()
    
    a = 2


if __name__ == "__main__":
    main()
