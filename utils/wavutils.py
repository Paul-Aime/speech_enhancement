import wave

import numpy as np
import torch
from scipy.io import wavfile

# from utils.cuda_utils import init_cuda

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def read_wav(filename, offset=0, nframes=None, dtype=torch.double):
    """Efficiently read wav files. Using `scipy.io.wavfile.read()` when
    reading the whole file, and `wave` library when reading a part of
    the file. Only implemented for mono channel sounds.

    Arguments:
        filename {str} -- wav file path

    Keyword Arguments:
        offset {int} -- First sample to read. (default: {0})
        nframes {int} -- Number of frames to be read. If None, then the
                         whole file is read. (default: {None})
        dtype {torch.dtype} -- Wanted torch data type for the output.
                               (default: {torch.double})

    Returns:
        torch.tensor(dtype) -- signal of shape torch.Size([1, L]) #! Assuming one channel
        int -- framerate
    """

    if nframes is None:  # Load whole file
        fs, x = wavfile.read(filename, mmap=False)
        x = torch.tensor(x, dtype=dtype)
        x.unsqueeze_(dim=0)

    else:  # Load a part
        with wave.open(filename) as f:
            fs = f.getframerate()
            f.setpos(offset)
            buff = f.readframes(nframes)
        x = torch.tensor(np.frombuffer(buff, np.int16), dtype=dtype)
        x.unsqueeze_(dim=0)
        x -= x.mean()

    return x.to(DEVICE), fs
