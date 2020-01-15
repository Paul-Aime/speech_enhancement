import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import wavfile

from model import dataset, net
from model.dataset import batchify
from utils import backup_utils, cuda_utils, params_utils
from utils.plot_utils import Font

# DEVICE = cuda_utils.init_cuda(verbose=True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')

EARLY_BREAK = False

###############################################################################
# Main


def main():

    verbose = True
    mode = 'auto'  # Load the last trained model from params settings

    params = params_utils.Params()
    params.snr = -5  # dB

    # Get the last from params settings
    if mode == 'auto':
        params.load_model = True
        model, _, chkpt_logs = net.get_model(params, verbose=verbose)
    # or load from a given checkpoint path
    else:
        chkpt_path = "./experiments/saved_models/park2017_R-CED9/fs8000_snr1_nfft256_hop128/010_0-000.pt"
        model = net.MyCNN(params)
        chkpt_logs = backup_utils.load_model(
            model, None, chkpt_path, verbose=True)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    test_set = dataset.CustomDataset(params.test_raw_csv_path,
                                     './data/noise/babble_test.wav', params, mode='test')

    test(model, loss_fn, test_set, params, save_out=True, verbose=True)


###############################################################################
# Main functions


def test(model, loss_fn, data_set, params, save_out=False, verbose=True):

    if verbose:
        start_t = datetime.datetime.now()

    model.eval()
    dataset_size = len(data_set)
    # Freeze model's params
    for p in model.parameters():
        p.requires_grad = False

    # To compute mean loss
    loss_hist = torch.zeros(dataset_size)
    len_hist = torch.zeros(dataset_size)

    # Early break for quick testing
    if EARLY_BREAK:
        mm = 0

    # Each sound is considered as a batch, keep only module
    for i, ((sig_noisy, sig_clean), (x_abs, x_ang), (y_abs, y_ang)) in enumerate(data_set.batch_loader()):

        if EARLY_BREAK:
            mm += 1
            if mm > 3:
                break

        # Get sound ID
        # for test and validaton mode, `data_set.snd_indices[i] == i`, but still
        sound_path_id = data_set.get_sound_path_id(data_set.snd_indices[i])

        # Batchify x
        X = batchify(x_abs, params.n_frames)  # shape (B, C, H, W)

        # Feed forward
        Y_pred = model(X)
        assert Y_pred.device == DEVICE

        # Go from batch to reconstructed STFT
        y_pred = Y_pred.squeeze().T.unsqueeze(0)

        # Compute loss
        loss = loss_fn(y_abs, y_pred)
        loss = loss.sum()

        # Backup loss
        loss_hist[i] = loss.data
        len_hist[i] = y_pred.shape[2]

        # Print info
        if verbose:
            print(sound_path_id)
            epoch_percent = ((i+1) / dataset_size) * 100
            elapsed_t = datetime.datetime.now() - start_t
            elapsed_t_str = '{:02.0f}:{:02.0f}  -- {:5.1f}%  #{:4d}/{:d}'.format(
                *divmod(elapsed_t.seconds, 60), epoch_percent, i+1, dataset_size)
            print("  loss (x1000): {:.6f} (elapsed: {})".format(
                loss.data*1000, elapsed_t_str), end='\n')

        # Reconstruct signal from STFT
        length = sig_clean.shape[-1]  # ensure y_pred has good length, or None
        sig_pred = reconstruct_signal(y_abs, y_ang, params, length=length)
        if not sig_pred.shape[-1] == sig_noisy.shape[-1] == sig_clean.shape[-1]:
            print('Reconstructed signal has not the same shape as original.')

        # Compute metrics
        metrics_names = ['snr_clean_vs_noisy',
                         'snr_clean_vs_pred',
                         'snr_pred_vs_noisy']
        metrics = compute_metrics(
            sig_clean, sig_noisy, sig_pred, metrics_names)

        # Save outputs
        if save_out:

            save_outputs((sig_clean, sig_noisy, sig_pred), 
                         (y_abs, x_abs, y_pred), 
                         sound_path_id, params,
                         modes=('clean', 'noisy', 'pred'))

            save_metrics(metrics, metrics_names, sound_path_id, params)

            # TODO also save losses

    # Print mean loss over all sounds
    loss_mean = torch.sum(loss_hist * len_hist) / len_hist.sum()
    if verbose:
        print("\n  mean loss (x1000): {:.6f}".format(loss.data*1000))

    return loss_mean, loss_hist, len_hist


###############################################################################
# Metrics

def compute_metrics(sig_clean, sig_noisy, sig_pred, metrics_names):

    # TODO retrieve noise added to create sig_noisy, because of normalizations...

    metrics = []
    for m_name in metrics_names:

        if m_name == 'snr_clean_vs_noisy':
            m = snr(sig_clean, sig_noisy)
        elif m_name == 'snr_clean_vs_pred':
            m = snr(sig_clean, sig_pred)
        elif m_name == 'snr_pred_vs_noisy':
            m = snr(sig_pred, sig_noisy)
        else:
            print('ERROR: metric unknown.')
            return

        metrics.append(m)

    return metrics


def snr(x, y, mode='dB'):
    x, y = x.squeeze(), y.squeeze()

    diff = x-y

    snr_ = ((x-x.mean())**2).sum() / ((diff-diff.mean())**2).sum()

    if mode.lower() == 'db':
        snr_ = 10 * torch.log10(snr_)

    return snr_


###############################################################################
# Saving functions

def save_outputs(signals, spectrograms, sound_path_id, params,
                 modes=('noisy', 'pred')):

    assert len(signals) == len(spectrograms) == len(modes)

    for mode, signal, spect in zip(modes, signals, spectrograms):

        spid_mode = sound_path_id + '_' + mode

        # Save signal, in '.wav' and '.signal' (tensor)
        signal_dir = params.signals_saving_dir
        signal_path = os.path.join(signal_dir, spid_mode)
        if not os.path.isdir(os.path.dirname(signal_path)):
            os.makedirs(os.path.dirname(signal_path))
        torch.save(signal, signal_path + '.signal')
        wavfile.write(signal_path + '.wav', int(params.fs),
                      signal.squeeze().cpu().numpy())

        # Save spectrogram
        spectrogram_dir = params.spectrograms_saving_dir
        spectrogram_path = os.path.join(spectrogram_dir, spid_mode)
        if not os.path.isdir(os.path.dirname(spectrogram_path)):
            os.makedirs(os.path.dirname(spectrogram_path))
        save_spectrogram(spect, spectrogram_path + '.png', params)
        torch.save(spect, spectrogram_path + '.spect')


def save_metrics(metrics, metrics_names, sound_path_id, params):
    # metrics and metrics_names are sequences
    assert len(metrics) == len(metrics_names)

    for metric, metric_name in zip(metrics, metrics_names):
        mt_sv_dir = params.metrics_saving_dir(metric_name)
        mt_sv_path = os.path.join(mt_sv_dir, sound_path_id + '.mt')

        if not os.path.isdir(os.path.dirname(mt_sv_path)):
            os.makedirs(os.path.dirname(mt_sv_path))

        torch.save(metric, mt_sv_path)


def save_spectrogram(spectrogram, saving_path, params):

    n_fft = params.n_fft
    fs = params.fs

    fig, ax = plt.subplots(nrows=1, ncols=1)
    # im = ax.imshow(spectrogram.squeeze(),
    #                origin='lower',
    #                cmap=plt.get_cmap('magma'))
    im = ax.pcolor(spectrogram.squeeze(),
                   cmap=plt.get_cmap('magma'),
                   vmin=0, vmax=1)
    ax.set_xlabel('STFT frame number',
                  fontproperties=Font().axis_labels, fontweight='bold')
    ax.set_ylabel('Frequencies', fontproperties=Font(
    ).axis_labels, fontweight='bold')

    # Custom yticks
    yticks_step_in_hz = 500
    yticks_hz = np.arange(0, fs/2+1, yticks_step_in_hz)
    yticks = yticks_hz * (n_fft/fs)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_hz.astype(np.int))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()

    if not os.path.isdir(os.path.dirname(saving_path)):
        os.makedirs(os.path.dirname(saving_path))

    plt.savefig(saving_path)

###############################################################################
# Functions


def reconstruct_signal(S_abs, S_ang, params, length=None):
    y = dataset.istft(S_abs, S_ang, length=length, **params.istft_kwargs)
    y = y.unsqueeze(0)
    y = dataset.normalize_sound(y, inplace=True)
    return y


###############################################################################
# Main

if __name__ == "__main__":
    main()
