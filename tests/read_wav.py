""" Testing libraries to find the fastest way to read a wav file

Librosa not here because definitely slower.

Conclusion :
- Use `scipy.io.wavfile.read()` when loading the whole file.
- Use `wave` + `np.from_buffer()` when loading a part.
"""
import wave

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.utils import time


ACTIVATED = True
NUMBER = 10000
PRINT_FUNC_CALL = True


def main():

    # Parameters
    filename = './tests/SA1.wav'
    offset = 2048
    nframes = 1024

    # Check for consistence
    if not ACTIVATED:

        # Loading whole file
        x1 = load1(filename)
        x2 = load2(filename)
        x3 = load3(filename)
        assert torch.all(torch.eq(x1, x2)) and torch.all(torch.eq(x2, x3))

        # Loading a part
        x1 = load1_part(filename, offset, nframes)
        x2 = load2_part(filename, offset, nframes)
        x3 = load3_part(filename, offset, nframes)
        assert torch.all(torch.eq(x1, x2)) and torch.all(torch.eq(x2, x3))

    # Time it
    if ACTIVATED:

        # Loading the whole file
        print('\n\n---Loading the whole file.')
        print('\n  #1 wave: ', end='')
        print(load1(filename))
        print('\n  #2 torchaudio: ', end='')
        print(load2(filename))
        print('\n  #3 scipy: ', end='')
        print(load3(filename))

        # Loading a part
        print('\n\n---Loading a part.')
        print('\n  #1 wave: ', end='')
        print(load1_part(filename, offset, nframes))
        print('\n  #2 torchaudio: ', end='')
        print(load2_part(filename, offset, nframes))
        print('\n  #3 scipy: ', end='')
        print(load3_part(filename, offset, nframes))


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load1(filename):
    # With wave + numpy.frombuffer
    with wave.open(filename) as f:
        fs = f.getframerate()
        buff = f.readframes(f.getnframes())
    x = torch.tensor(np.frombuffer(buff, np.int16), dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load2(filename):
    # With torchaudio.load_wav
    x, fs = torchaudio.load_wav(filename)
    x = x.to(dtype=torch.double)

    return x


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load3(filename):
    # With scipy.io.wavfile.read
    fs, x = wavfile.read(filename, mmap=False)
    x = torch.tensor(x, dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load1_part(filename, offset, nframes):
    # With wave + numpy.frombuffer
    with wave.open(filename) as f:
        fs = f.getframerate()
        f.setpos(offset)
        buff = f.readframes(nframes)
    x = torch.tensor(np.frombuffer(buff, np.int16), dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load2_part(filename, offset, nframes):
    # With torchaudio.load_wav
    x, fs = torchaudio.load_wav(filename)
    x = x[:, offset:offset+nframes].to(dtype=torch.double)

    return x


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def load3_part(filename, offset, nframes):
    # With scipy.io.wavfile.read
    fs, x = wavfile.read(filename, mmap=True)
    x = torch.tensor(x[offset:offset+nframes], dtype=torch.double)
    x.unsqueeze_(dim=0)

    return x


if __name__ == '__main__':
    main()
