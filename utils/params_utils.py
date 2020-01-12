import os
from pprint import pformat

import torch
import numpy as np


class Params():

    def __init__(self):

        # --- Learning parameters
        self.learning_rate = 1e-3
        self.n_frames = 7
        self.max_epoch = 50
        # same ratio as noise set ratio (in seconds)
        self.train_val_ratio = 180/(180+25)

        # --- Dataset parameters

        # Sounds
        self.fs = 8*1e3  # 8kHz
        self.snr = 1  # in dB

        # STFT
        self.n_fft = 256
        self.hop_length = self.n_fft // 2
        # 'hann', # TODO make a string, to be accessible from id_dict
        self.window = torch.hann_window(self.n_fft).numpy()

        # --- Paths

        # Data
        self.data_root = './data'
        self.noise_dirname = 'noise/'
        self.raw_dirname = 'raw/'
        self.train_raw_csv_name = 'train_raw.csv'
        self.test_raw_csv_name = 'test_raw.csv'
        self.train_noise_csv_name = 'train_noise.csv'
        self.val_noise_csv_name = 'val_noise.csv'
        self.test_noise_csv_name = 'test_noise.csv'

        # Backup
        self.save_model = True
        self.load_model = False
        self.backup_root = './experiments/saved_models'
        self.backup_saving_dir = 'park2017_R-CED9'
        
        self.__init_id_dict()

    # ------------------------------------------------------------------
    #

    @property
    def stft_kwargs(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": None,
            "window": self.window,
            "center": True,
            "dtype": np.complex64,
            "pad_mode": 'reflect'
        }

    # ------------------------------------------------------------------
    # Dictionnary

    def __init_id_dict(self):

        self._id_dict_keys = ('fs',
                              'snr',
                              'n_fft',
                              'hop_length')

    @property
    def id_dict(self):
        return {k: self.__dict__[k] for k in self._id_dict_keys}

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        return pformat(self.id_dict)

    # ------------------------------------------------------------------
    # Paths

    # --- Model ID

    @property
    def model_id(self):
        return "fs{fs:04.0f}_snr{snr}_nfft{n_fft}_hop{hop_length}"\
            .format(**self.id_dict)

    # --- Input paths

    @property
    def train_raw_csv_path(self):
        return os.path.join(self.data_root, self.train_raw_csv_name)

    @property
    def test_raw_csv_path(self):
        return os.path.join(self.data_root, self.test_raw_csv_name)

    @property
    def train_noise_csv_path(self):
        return os.path.join(self.data_root, self.train_noise_csv_name)

    @property
    def val_noise_csv_path(self):
        return os.path.join(self.data_root, self.val_noise_csv_name)

    @property
    def test_noise_csv_path(self):
        return os.path.join(self.data_root, self.test_noise_csv_name)

    # --- Backup

    @property
    def model_saving_dir(self):
        return os.path.join(self.backup_root, self.backup_saving_dir, self.model_id)

    @property
    def best_model_copy_dir(self):
        return os.path.join(self.backup_root, 'best', self.backup_saving_dir)

    @property
    def best_model_copy_path(self):
        return os.path.join(self.best_model_copy_dir, self.model_id + '.pt')

    @property
    def last_model_copy_dir(self):
        return os.path.join(self.backup_root, 'last', self.backup_saving_dir)

    @property
    def last_model_copy_path(self):
        return os.path.join(self.last_model_copy_dir, self.model_id + '.pt')

    @property
    def pred_dir(self):
        return os.path.join(self.data_root, self.pred_dir,
                            self.backup_saving_dir, self.model_id)

    def pred_path(self, sound_name, ext='.pred'):
        return os.path.join(self.pred_dir, sound_name + ext)

    # TODO prediction will have a spectrogram and a time folder
    # TODO as noise is added on the fly, we also need to register the corresponding noisy data
