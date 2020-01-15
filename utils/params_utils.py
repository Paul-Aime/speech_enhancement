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

        # Backup
        self.experiments_root = './experiments'
        self.experiment_name = 'park2017_R-CED9'

        self.save_model = True
        self.load_model = False

        self.saved_models_dirname = 'saved_models'
        self.spectrograms_dirname = 'spectrograms'
        self.signals_dirname = 'signals'
        self.metrics_dirname = 'metrics'

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

    @property
    def istft_kwargs(self):
        stft_kwargs = self.stft_kwargs
        return {
            "hop_length": stft_kwargs['hop_length'],
            "win_length": stft_kwargs['win_length'],
            "window": stft_kwargs['window'],
            "center": stft_kwargs['center']
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

    # --- Backup

    # - Roots
    @property
    def saved_models_root(self):
        return os.path.join(self.experiments_root, self.saved_models_dirname)

    @property
    def spectrograms_root(self):
        return os.path.join(self.experiments_root, self.spectrograms_dirname)

    @property
    def signals_root(self):
        return os.path.join(self.experiments_root, self.signals_dirname)

    @property
    def metrics_root(self):
        return os.path.join(self.experiments_root, self.metrics_dirname)

    # - With params suffix
    @property
    def model_saving_dir(self):  # TODO model <- models
        return os.path.join(self.saved_models_root, self.experiment_name, self.model_id)

    @property
    def signals_saving_dir(self):
        return os.path.join(self.signals_root, self.experiment_name, self.model_id)

    @property
    def spectrograms_saving_dir(self):
        return os.path.join(self.spectrograms_root, self.experiment_name, self.model_id)

    def metrics_saving_dir(self, metric_name):
        return os.path.join(self.metrics_root, metric_name, self.experiment_name, self.model_id)

    # - With actual name (methods)
    # TODO to replace `backup_utils.get_model_saving_path`
    # def model_saving_path(self, epoch, loss):
    #     loss_str = "".join([l if l != '.' else '-'
    #                         for l in "{:.3f}".format(loss)])
    #     return os.path.join(params.model_saving_dir, '{:03d}_{}.pt'.format(epoch, loss_str))

    # May not be useful
    def signal_saving_path(self, sound_path_id, ext='.wav'):
        return os.path.join(self.signals_saving_dir, sound_path_id + ext)

    # May not be useful
    def spectrogram_saving_path(self, sound_path_id, ext='.png'):
        return os.path.join(self.spectrograms_saving_dir, sound_path_id + ext)

    # May not be useful
    def metric_saving_path(self, metric_name, sound_path_id):
        return os.path.join(self.metrics_saving_dir(metric_name), sound_path_id)

    # --- Output backup
    # @property
    # def spectrograms:
    #     pass

    # TODO prediction will have a spectrogram and a time folder
    # TODO as noise is added on the fly, we also need to register the corresponding noisy data
