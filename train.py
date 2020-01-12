import datetime
import copy

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

    params = params_utils.Params()

    # --- Model
    model = net.MyCNN(params)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # Print model's state_dict
    if verbose:
        print("\nModel's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t\t", model.state_dict()
                  [param_tensor].size())

    # Load a checkpoint
    chkpt_logs = None
    if params.load_model:
        saved_model_path = "./experiments/saved_models/experiment1/fs8000_snr1_nfft256_hop128/003_0-618.pt"
        chkpt_logs = backup_utils.load_checkpoint(
            model, optimizer, saved_model_path)
        print('\nModel parameters updated with saved model :')
        print('  ' + saved_model_path)

    # Datasets
    train_set = dataset.CustomDataset(params.train_raw_csv_path,
                                      './data/noise/babble_train.wav', params, mode='train')
    val_set = dataset.CustomDataset(params.train_raw_csv_path,
                                    './data/noise/babble_val.wav', params, mode='validation')
    # test_set = dataset.CustomDataset(params.test_raw_csv_path,
    #                                  params.test_noise_csv_path, params, mode='test')

    train(model, optimizer, loss_fn, train_set, val_set, params,
          chkpt_logs=chkpt_logs)


###############################################################################
# Main functions


def train(model, optimizer, loss_fn, train_set, val_set, params,
          chkpt_logs=None, verbose=True):

    logs = TrainingHistory()
    if chkpt_logs is not None:
        logs.load_from_other(chkpt_logs)

    nn = 0  # Early break for testing
    while not logs.early_stop:

        nn += 1  # TODO withdraw early break
        if nn > 3:
            break

        if verbose:
            start_t = datetime.datetime.now()
            print('\n' + '='*50)
            # logs.epoch not yet up-to-date, hence `+1`
            print("epoch #{}".format(logs.epoch + 1))

        for mode, data_set in zip(('train', '  val'), (train_set, val_set)):

            if mode == 'train':
                model.train()
                dataset_size = data_set.train_size
                # Unfreeze model's params
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.eval()
                dataset_size = data_set.val_size
                # Freeze model's params
                for p in model.parameters():
                    p.requires_grad = False

            # Compute verbose step
            if verbose:
                verb_step = (dataset_size // 100)+1

            # To compute mean loss over the full batch
            loss_hist = torch.zeros(dataset_size)
            len_hist = torch.zeros(dataset_size)

            # Each sound is considered as a batch, keep only module
            # mm = 0  # early break for testing
            for i, ((x, _), (y, _)) in enumerate(data_set.batch_loader()):

                # mm += 1  # TODO withdraw early break
                # if mm > 3:
                #     break

                # Batchify x
                X = batchify(x, params.n_frames)  # shape (B, C, H, W)

                # shape of X : (B, C, H, W), where :
                # B is the number of frames of the STFT of the sound
                # C is the number of channels (equals to 1 in our case)
                # H is the height of each input sample (equals to params.nfft//2 +1)
                # W is the width of each input sample (equals to params.n_frames)

                # Feed forward
                Y_pred = model(X)
                assert Y_pred.device == DEVICE

                # Go from batch to reconstructed STFT
                # y_pred = reconstruct(Y_pred)
                y_pred = Y_pred.squeeze().T.unsqueeze(0)  # TODO optimize

                # Compute loss
                loss = loss_fn(y, y_pred)
                loss = loss.sum()

                # Learn
                if mode == 'train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Backup loss
                loss_hist[i] = loss.data
                len_hist[i] = y_pred.shape[2]

                # Print info (or register loss_mean if last sound)
                if (verbose and not i % verb_step) or (i+1 == dataset_size):
                    loss_mean = torch.sum(loss_hist[: i+1] *
                                          len_hist[:i+1]) / len_hist[:i+1].sum()

                    epoch_percent = ((i+1) / dataset_size) * 100
                    elapsed_t = datetime.datetime.now() - start_t
                    elapsed_t_str = '{:02.0f}:{:02.0f}  -- {:5.1f}%  #{:4d}/{:d}'.format(
                        *divmod(elapsed_t.seconds, 60), epoch_percent, i+1, dataset_size)
                    print("  {} loss: {:.3f} (elapsed: {})".format(
                        mode, loss_mean, elapsed_t_str), end='\r')

                # Register loss
                if i + 1 == dataset_size:
                # if True:  # When there are early stops for testing # TODO withdraw early break
                    if mode == 'train':
                        train_loss = loss_mean
                    else:
                        val_loss = loss_mean

            # Re-print info, but not to be erased
            if verbose:
                print("  {} loss: {:.3f} (elapsed: {})".format(
                    mode, loss_mean, elapsed_t_str))

        # --- Save model

        # Get the path
        if not params.save_model:
            saved_model_path = None
        else:
            # 'logs' are not yet up-to-date, hence '+1'
            saved_model_path = backup_utils.get_model_saving_path(
                logs.epoch + 1, train_loss, params)

        # Update logs
        logs.add_values(train_loss, val_loss, saved_model_path)

        # Save model
        if params.save_model:
            assert saved_model_path == backup_utils.save_checkpoint(
                model, optimizer, train_loss, logs, params)

        if logs.epoch >= params.max_epoch:
            break

    return logs


###############################################################################
# Classes

class TrainingHistory():

    def __init__(self):

        self.__epoch = 0
        self.__saved_models_paths = []

        self.__train_loss = []
        self.__val_loss = []

        self.patience = 10
        self.margin = 0.005

    # ---------------------------------------------------------------- #
    @property
    def best_model_path(self):

        if not self.early_stop:
            print('WARNING: Training unfinished.')

        idx_bst = np.argmin(self.val_loss)

        return self.saved_models_paths[idx_bst]

    @property
    def best_model_val_loss(self):
        return np.min(self.val_loss)

    @property
    def epoch(self):
        return self.__epoch

    @property
    def saved_models_paths(self):
        return self.__saved_models_paths

    @property
    def early_stop(self):

        if len(self.val_loss) <= self.patience:
            return False

        val_loss_min = min(self.val_loss)
        val_loss_late_min = min(self.val_loss[-self.patience:])

        return bool(val_loss_late_min > ((1 + self.margin) * val_loss_min))

    @property
    def train_loss(self):
        return self.__train_loss

    @property
    def val_loss(self):
        return self.__val_loss

    # ---------------------------------------------------------------- #
    def load_from_other(self, other_dict):

        for k, v in other_dict.items():
            self.__dict__[k] = copy.deepcopy(v)

    def add_values(self, train_loss, val_loss, model_path):

        train_loss = float(train_loss)
        val_loss = float(val_loss)

        self.__epoch += 1
        self.__saved_models_paths.append(model_path)

        self.__train_loss.append(train_loss)
        self.__val_loss.append(val_loss)

    # ------------------------------------------------------------------

    def plot_loss(self):
        # TODO label legends, autosave etc
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)


if __name__ == "__main__":
    main()
