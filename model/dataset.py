import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from librosa.core import istft as librosa_istft
from librosa.core import stft as librosa_stft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

# For imports to work from __main__ and from other folders
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(1, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

if True:  # For the order to not be broken with autoformatter
    from utils import cuda_utils
    from utils.utils import sec_to_hms
    from utils.wavutils import read_wav
    # from utils.cuda_utils import init_cuda


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

##################################################
# Main


def main():

    # --- Paths
    root_dir = './data/raw'

    csv_raw_train = './data/train_raw.csv'
    csv_raw_test = './data/test_raw.csv'

    csv_noise_train = './data/train_noise.csv'
    csv_noise_val = './data/val_noise.csv'
    csv_noise_test = './data/test_noise.csv'

    noise_path_train = './data/noise/babble_train.wav'
    noise_path_val = './data/noise/babble_val.wav'
    noise_path_test = './data/noise/babble_test.wav'

    # --- Parameters
    fs = 8 * 1e3  # 8 kHz
    snr = 1       # in dB
    n_fft = 256
    hop_length = n_fft // 2
    stft_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": None,
        "window": torch.hann_window(n_fft).numpy(),  # 'hann',
        "center": True,
        "dtype": np.complex64,
        "pad_mode": 'reflect'  # 'reflect'
    }

    # Check if csv files are done
    if not all(os.path.exists(f) for f in (csv_raw_train, csv_raw_test)):
        create_csv(root_dir, train_path=csv_raw_train, test_path=csv_raw_test)

    # TODO separate indices from train and val from timit_train folder
    train_set = CustomDataset(root_dir, csv_raw_train,
                              noise_path_train, fs, snr, stft_kwargs)
    # val_set = CustomDataset(root_dir, csv_raw_train,
    #                         noise_path_val, fs, snr, stft_kwargs)
    # test_set = CustomDataset(root_dir, csv_raw_train,
    #                          noise_path_test, fs, snr, stft_kwargs)

    # Plot histograms of lengths
    # _, ax = plt.subplots()
    # train_set.histogram_wav_length(ax, label='Train')
    # test_set.histogram_wav_length(ax, label='Test')
    # ax.legend()
    # plt.show()

    # Get an item and plot it
    print('Taille dataset', len(train_set))
    x, y = train_set[56]
    x = x[0] ** 2  # keep only module
    y = y[0] ** 2  # keep only module

    #wave.write('./tests/test1.wav', 16000, x)
    #torchaudio.save('./tests/test1.wav', y, 16000)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    spect1 = ax1.imshow(x.squeeze())
    ax1.set_title('input : {:d}, {:d}'.format(*x.squeeze().shape))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(spect1, cax=cax, orientation='vertical')

    spect2 = ax2.imshow(y.squeeze())
    ax2.set_title('ground truth : {:d}, {:d}'.format(*y.squeeze().shape))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(spect2, cax=cax, orientation='vertical')
    plt.show()

##################################################
# Classes


class CustomDataset(Dataset):
    """TIMIT dataset."""

    def __init__(self, csv_raw, noise_path, params, mode='test'):

        self.mode = mode

        # Paths
        self.root_dir = params.data_root
        self.csv_raw = csv_raw
        self.raw_paths = pd.read_csv(self.csv_raw).to_numpy().squeeze()
        self.noise_path = noise_path

        # Parameters
        self.fs = params.fs
        self.snr = params.snr
        self.stft_kwargs = params.stft_kwargs
        self.params = params

        # Sound indices for the given mode
        if mode != 'test':
            self.train_size = int(self.params.train_val_ratio * len(self))
            self.val_size = len(self) - self.train_size

            if self.mode == 'train':
                self.snd_indices = np.arange(self.train_size)
            elif self.mode == 'validation':
                self.snd_indices = np.arange(self.train_size, len(self))
            else:
                return ('ERROR : unknown mode, must be one of str(test, train, validation)')

        elif self.mode == 'test':
            self.snd_indices = np.arange(len(self))
        else:
            return ('ERROR : unknown mode, must be one of str(test, train, validation)')

        # Resampler
        self.resampler = torchaudio.transforms.Resample(new_freq=self.fs)

        # Noise saved in RAM
        self.noise, fs_orig = read_wav(self.noise_path)
        self.noise = normalize_sound(self.noise)
        self.resampler.orig_freq = fs_orig
        # TODO why float needed ?
        self.noise = self.resampler(
            self.noise.float().cpu()).to(DEVICE).double()
        self.noise = normalize_sound(self.noise)
        self.noise_len = self.noise.shape[1]
        self.noise_len_in_s = self.noise_len * (1/self.fs)

    # ------------------------------------------------------------------
    # Magic methods

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):

        # Read raw audio ground truth
        raw_target, fs_orig = read_wav(self.raw_paths[index])
        self.resampler.orig_freq = fs_orig
        # TODO why it's needed to go with float ...
        raw_target = self.resampler(
            raw_target.float().cpu()).to(DEVICE).double()
        raw_target = normalize_sound(raw_target)

        # Add noise to the raw ground truth to create the raw input
        raw_in = self.add_noise(raw_target)
        # raw_in = normalize_sound(raw_in)
        # TODO
        # Ne pas normaliser le signal bruité, ni la stft correspondante,
        # car en faisant ainsi on diminue donc l'amplitude du signal
        # clean, d'autant plus qu'on ajoute du bruit, donc on ne peut
        # pas comparer le pred avec le clean sans renormaliser le pred,
        # et donc au final on perd toute possibilité de comparé la perte
        # gloable d'amplitude, aka les faux positifs
        # (débruitage alors qu'il fallait pas)

        # Convert to time-frequency domain using STFT
        x_abs, x_ang = stft(raw_in, **self.stft_kwargs)  # tuple (x_abs, x_ang)
        y_abs, y_ang = stft(raw_target, **self.stft_kwargs)  # tuple (y_abs, y_ang)
        
        x_abs = normalize_stft(x_abs, mode='max')
        x_ang = normalize_stft(x_ang, mode='max')
        
        y_abs = normalize_stft(y_abs, mode='max')
        y_ang = normalize_stft(y_ang, mode='max')

        assert x_abs.device == DEVICE

        return (raw_in, raw_target), (x_abs, x_ang), (y_abs, y_ang)

    # ------------------------------------------------------------------
    # Dataloader utilities

    def batch_loader(self):

        if self.mode == 'train':
            self.snd_indices = np.random.permutation(self.snd_indices)

        for snd_id in self.snd_indices:
            yield self[snd_id]  # a batch is a full sound

    def get_sound_path_id(self, idx):
        sound_path = self.raw_paths[idx]
        # Remove base folder # TODO do it better than that hardcoded 3
        sound_path = os.path.join(*sound_path.split('/')[-3:])
        # Remove extension
        return os.path.splitext(sound_path)[0]

    # ------------------------------------------------------------------
    # Dataset utilities

    def gen_noise(self, nb_samples):
        """Randomly select a part of the noise

        Arguments:
            nb_samples {int} -- length of the wanted sequence in number
                                of samples

        Returns:
            torch.tensor(dtype=torch.double) -- sequence of noise.
                                                shape torch.Size([nb_samples])
        """

        # TODO may "need" to add + 1 to second argument
        idx = np.random.randint(0, self.noise_len - nb_samples)

        return self.noise[:, idx: idx+nb_samples]

    def add_noise(self, sig):
        # `sig` size must be (C, T)
        noise = self.gen_noise(nb_samples=sig.shape[1])
        return add_noise_snr(sig, noise, self.snr)

    # ------------------------------------------------------------------
    # Exploration utilities

    def histogram_wav_length(self, ax=None, label=None):

        print('Computing histogram of lengths.')
        rates = np.zeros(len(self))
        n_samples = np.zeros(len(self))
        for i, wav_path in enumerate(self.raw_paths):
            if not i % 100:
                print('#{:03d}/{:d}'.format(i+1, len(self)), end='\r')
            si, _ = torchaudio.info(wav_path)
            rates[i], n_samples[i] = si.rate, si.length

        rate = rates[0]
        if not all(r == rate for r in rates):
            print('WARNING : not all rates are the same')
            return

        if ax is None:
            _, ax = plt.subplots()
        n, _, p = ax.hist(n_samples, bins=50)
        ax.set_title('Histogramme des durées')
        xticks = ax.get_xticks()
        ax.set_xticklabels(["{:.0f}\n{:.2f}".format(t, t/rate)
                            for t in xticks])
        ax.set_xlabel("Nombre d'échantillons [] | Durée [s]")

        if label is not None:
            p[0].set_label('{} ({:d} samples | {:d}h{:02d}min{:02.0f}s)'.format(
                label, int(sum(n_samples)), *sec_to_hms(sum(n_samples)/rate)))


##################################################
# Dataset utilities

def batchify(x, nframes):

    # shape of x : (C, H, W)

    # Add padding to the left and the right
    x = pad(x, (nframes-1)//2)

    # Width of the STFT, equals to the total number of frames
    w = x.shape[2]

    X = torch.stack(tuple(x[0, :, i:i+nframes]
                          for i in range(w - nframes + 1)),
                    0).unsqueeze(dim=1)  # Unsqueeze for the channel dim

    return X


def pad(x, nframes):
    # padding nframes to the left and the rigth, by simple replication
    # shape of x : (C, H, W)

    return torch.cat((x[:, :, :nframes], x, x[:, :, -nframes:]), dim=2)


##################################################
# Sounds


def stft(x, **kwargs):
    """
    Only for 1xL tensors, i.e. C = 1
    https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa-core-stft
    
    S_abs is the magnitude of frequency bin f at frame t
    
    The integers t and f can be converted to physical units by means of
    the utility functions frames_to_sample and fft_frequencies.
    
    """
    S = librosa_stft(x.squeeze().cpu().numpy(), **kwargs)
    S_abs = torch.tensor(np.abs(S), dtype=torch.double,
                         device=DEVICE).unsqueeze(dim=0)
    S_ang = torch.tensor(np.angle(S), dtype=torch.double,
                         device=DEVICE).unsqueeze(dim=0)

    return S_abs, S_ang


def istft(S_module, S_angle, length=None, **kwargs):
    """
    Only for 1xL tensors, i.e. C = 1
    https://librosa.github.io/librosa/generated/librosa.core.istft.html#librosa.core.istft
    """
    S_module = S_module.squeeze().numpy()
    S_angle = S_angle.squeeze().numpy()
    return librosa_istft(S_module + 1j * S_angle, length=length, **kwargs)


def add_noise_snr(sig, noise, snr):
    """ shape [CxL] channel x length
    Despite it being a bit less readable than one can do, it is faster that way

    """
    # Center channels
    sig.sub_(sig.mean(dim=1).unsqueeze(0).T)
    noise.sub_(noise.mean(dim=1).unsqueeze(0).T)

    # Calcul addition coefficient
    alpha = ((torch.mean(sig**2, dim=1) /  # power of sig
              torch.mean(noise**2, dim=1)) * (10 ** (-snr/10))).sqrt()

    return sig + (noise.T * alpha).T


def normalize_sound(x, inplace=True):
    "Shape CxL"

    if not inplace:
        x = x.clone()

    # Handle 1D vectors
    # if len(x.shape) == 1:
    #     x = x.unsqueeze(dim=0)

    # Center channles
    x.sub_(x.mean(dim=1).unsqueeze(dim=0).T)

    # Divide by maximum magnitude to put x in range [-1 1]
    x.div_(x.abs().max(dim=1).values.unsqueeze(dim=0).T)

    # Handle 1D vectors
    # if len(x.shape) == 1:
    #     x = x.squeeze()

    return x


def normalize_sound2(x):
    "Only for 1 channel"
    x -= x.mean()
    mag_max = max(-x.min(), x.max())
    return x / mag_max


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


##################################################
# Files

def create_csv(root_dir, train_path='./train_raw.csv', test_path='./test_raw.csv'):
    """Create a csv file for TIMIT corpus.

    Arguments:
        root_dir {str} -- root directory from which create_csv will search for raw data

    Keyword Arguments:
        train_path {str} -- output csv filepath for train data (default: {'./train_raw.csv'})
        test_path {str} -- output csv filepath for test data (default: {'./test_raw.csv'})

    Returns:
        tuple(str, str) -- Tuple (train_path, test_path) with created csv filepaths
    """

    wav_paths_train = []
    wav_paths_test = []

    for root, _, files in os.walk(root_dir):

        if ('train' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_train += [os.path.join(root, f) for f in wav_files]

        if ('test' in root) and (files is not None) and any(f.endswith('.wav') for f in files):
            wav_files = [f for f in files if f.endswith('.wav')]
            wav_paths_test += [os.path.join(root, f) for f in wav_files]

    pd.DataFrame(data={"col1": wav_paths_train}).to_csv(train_path,
                                                        header=None, index=False)
    pd.DataFrame(data={"col1": wav_paths_test}).to_csv(test_path,
                                                       header=None, index=False)

    return train_path, test_path


##################################################
# Main

if __name__ == '__main__':
    main()
