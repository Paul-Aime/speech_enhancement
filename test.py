import datetime
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import backup_utils, params_utils, cuda_utils
from model import net, dataset
from model.dataset import batchify

# DEVICE = cuda_utils.init_cuda(verbose=True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')


###############################################################################
# Main


def main():

    verbose = True
    mode = 'auto'  # Load the last trained model from params settings

    params = params_utils.Params()

    # Get the last from params settings
    if mode == 'auto':
        params.load_model = True
        model, optimizer, chkpt_logs = net.get_model(params, verbose=verbose)
    # or load from a given checkpoint path
    else:
        chkpt_path = "./experiments/saved_models/park2017_R-CED9/fs8000_snr1_nfft256_hop128/010_0-000.pt"
        model = net.MyCNN(params)
        chkpt_logs = backup_utils.load_model(
            model, None, chkpt_path, verbose=True)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    test_set = dataset.CustomDataset(params.test_raw_csv_path,
                                     './data/noise/babble_test.wav', params, mode='test')

    test(model, loss_fn, test_set, params, verbose=True)


###############################################################################
# Main functions


def test(model, loss_fn, data_set, params, verbose=True):

    # nn += 1  # TODO withdraw early break
    # if nn > 3:
    #     break

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

    # Each sound is considered as a batch, keep only module
    # mm = 0  # early break for testing
    for i, ((x_abs, x_ang), (y_abs, y_ang)) in enumerate(data_set.batch_loader()):

        # mm += 1  # TODO withdraw early break
        # if mm > 3:
        #     break

        # Get sound ID
        sound_path = data_set.raw_paths[i]
        sound_path_id = os.path.join(*sound_path.split('/')[-3:])

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

    # Print mean loss over all sounds
    loss_mean = torch.sum(loss_hist * len_hist) / len_hist.sum()
    if verbose:
        print("\n  mean loss (x1000): {:.6f}".format(loss.data*1000))

    return loss_mean, loss_hist, len_hist


###############################################################################
# Classes

if __name__ == "__main__":
    main()
