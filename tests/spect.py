"""Benchmark consistency and speed between various spectrogram / stft
implementation

Conclusions (tmp):
- No consistency found, neither for values nor size of outputs
- torchaudio object the slowest
- torchaudio functional the quickest
- others similar
"""


import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.transforms.functional import normalize
from torchaudio.transforms import Spectrogram as torchSpectrogram
from torchaudio.functional import spectrogram as torch_spectrogram
from torchaudio.compliance.kaldi import spectrogram as torch_kaldi_spectrogram

from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal import stft as scipy_stft

from librosa.core import stft as librosa_stft


if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    # https://stackoverflow.com/a/27876800/10076676
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.utils import time
    from utils.wavutils import read_wav

ACTIVATED = False
NUMBER = 100


def main():

    # --- Settings
    filename = './tests/SA1.wav'
    x, fs = read_wav(filename)
    offset = 11000
    x = x[:, offset:offset+8000]
    n_fft = 256
    hop_length = 32
    n_overlap = n_fft - hop_length

    # --- Arguments

    # torchaudio.transforms.Spectrogram
    spec_torch_kwargs = {
        # Size of FFT, creates n_fft // 2 + 1 bins
        "n_fft": n_fft,
        # Window size. (Default: n_fft)
        "win_length": None,
        # Length of hop between STFT windows. ( Default: win_length // 2)
        "hop_length": hop_length,
        # Two sided padding of signal. (Default: 0)
        "pad": 0,
        # A fn to create a window tensor that is applied/multiplied to each frame/window. (Default: torch.hann_window)
        "window_fn": torch.hann_window,
        # Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: 2)
        "power": 1,
        # Whether to normalize by magnitude after stft. (Default: False)
        "normalized": True,
        # Arguments for window function. (Default: None)
        "wkwargs": None
    }
    
    # assert spect.shape[2] == wavform.shape[1] // hop_length + 1

    # torchaudio.functional.spectrogram
    spec_torch_func_kwargs = {
        "pad": 0,
        "window": torch.hann_window(n_fft),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "power": 1,
        "normalized": True
    }

    # torchaudio.compliance.kaldi.spectrogram
    spec_torch_kaldi_kwargs = {
        "blackman_coeff": 0.42,
        "channel": -1,
        "dither": 1.0,
        "energy_floor": 0.0,
        "frame_length": (1/fs) * n_fft / 0.001,  # milliseconds
        "frame_shift": (1/fs) * hop_length / 0.001,  # milliseconds
        "min_duration": 0.0,
        "preemphasis_coefficient": 0.97,
        "raw_energy": True,
        "remove_dc_offset": True,
        "round_to_power_of_two": True,
        "sample_frequency": fs,
        "snip_edges": False,  # pad sound edges
        "subtract_mean": False,  # False
        "window_type": 'hanning'
    }

    # scipy.signal.spectrogram
    spec_scipy_kwargs = {
        "fs": fs,
        "window": torch.hann_window(n_fft).numpy(),  # 'hanning',
        "nperseg": n_fft,
        "noverlap": n_overlap,
        "nfft": None,
        "detrend": 'constant',  # 'constant', False
        "return_onesided": True,
        "scaling": 'spectrum',  # 'density', 'spectrum'
        "axis": -1,
        "mode": 'magnitude'  # 'psd'
    }

    # scipy.signal.stft
    spec_scipy_stft_kwargs = {
        "fs": fs,
        "window": torch.hann_window(n_fft).numpy(),  # 'hann',
        "nperseg": n_fft,
        "noverlap": n_overlap,
        "nfft": None,
        "detrend": 'constant',  # 'constant', False
        "return_onesided": True,
        "boundary": 'zeros',  # 'zeros', None
        "padded": True,
        "axis": -1
    }

    # librosa.core.stft
    spec_librosa_stft_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": None,
        "window": torch.hann_window(n_fft).numpy(),  # 'hann',
        "center": True,
        "dtype": np.complex64,
        "pad_mode": 'reflect'  # 'reflect'
    }

    # --- Containers

    functions = [
        spec_torch,
        spec_torch_func,
        spec_torch_kaldi,
        spec_scipy,
        spec_scipy_stft,
        spec_librosa_stft
    ]

    funcnames = [
        'spec_torch',
        'spec_torch_func',
        'spec_torch_kaldi',
        'spec_scipy',
        'spec_scipy_stft',
        'spec_librosa_stft'
    ]  # Can't access from func.__name__ because of decorators

    arguments = [
        spec_torch_kwargs,
        spec_torch_func_kwargs,
        spec_torch_kaldi_kwargs,
        spec_scipy_kwargs,
        spec_scipy_stft_kwargs,
        spec_librosa_stft_kwargs
    ]

    if not ACTIVATED:  # Not timing
        # --- Compute spectrograms

        spectrograms = []
        for fn, kwargs in zip(functions, arguments):
            spectrograms.append(fn(x, **kwargs))

        # --- Check for consistency
        for funcname, spec in zip(funcnames, spectrograms):
            print('\n# --- {}:'.format(funcname))
            print(spec.shape)
            print("(min, max, mean, std) = ({:.5f}, {:.5f}, {:.5f}, {:.5f})".format(
                spec.min(), spec.max(), spec.mean(), spec.std()))

        # --- Plot
        fig, axes = plt.subplots(nrows=2, ncols=3)
        for funcname, spec, ax in zip(funcnames, spectrograms, axes.flatten()):
            im = ax.imshow(spec.squeeze())
            ax.set_title(funcname)
            fig.colorbar(im, ax=ax)
        plt.show()

    else:
        for fn, funcname, kwargs in zip(functions, funcnames, arguments):
            print('\n{}:'.format(funcname))
            print(fn(x, **kwargs))

@time(NUMBER, ACTIVATED)
def spec_torch(x, **kwargs):
    """https://pytorch.org/audio/transforms.html#spectrogram"""
    S = torchSpectrogram(**kwargs)(x)
    return normalize(S, (S.mean(),), (S.std(),))


@time(NUMBER, ACTIVATED)
def spec_torch_func(x, **kwargs):
    """https://pytorch.org/audio/functional.html#spectrogram"""
    S = torch_spectrogram(x, **kwargs)
    return normalize(S, (S.mean(),), (S.std(),))


@time(NUMBER, ACTIVATED)
def spec_torch_kaldi(x, **kwargs):
    """https://pytorch.org/audio/compliance.kaldi.html#spectrogram"""
    x = x.to(dtype=torch.float)
    S = torch_kaldi_spectrogram(
        x, **kwargs).to(dtype=torch.double).t().unsqueeze_(dim=0)
    return normalize(S, (S.mean(),), (S.std(),))


@time(NUMBER, ACTIVATED)
def spec_scipy(x, **kwargs):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html#scipy-signal-spectrogram"""
    S = torch.tensor(scipy_spectrogram(x, **kwargs)[2], dtype=torch.double)
    return normalize(S, (S.mean(),), (S.std(),))


@time(NUMBER, ACTIVATED)
def spec_scipy_stft(x, **kwargs):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html#scipy-signal-stft"""
    S = torch.tensor(np.abs(scipy_stft(x, **kwargs)[2]), dtype=torch.double)
    return normalize(S, (S.mean(),), (S.std(),))


@time(NUMBER, ACTIVATED)
def spec_librosa_stft(x, **kwargs):
    """https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa-core-stft"""
    S = torch.tensor(np.abs(librosa_stft(x[0].cpu().numpy(), **kwargs)),
                     dtype=torch.double).unsqueeze_(dim=0)
    return normalize(S, (S.mean(),), (S.std(),))


if __name__ == '__main__':
    main()
