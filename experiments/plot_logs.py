import os
import torch
import numpy as np
import matplotlib.pyplot as plt

if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.plot_utils import Font
    from utils.params_utils import Params


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def main():

    params = Params()
    saved_models_root = params.saved_models_root

    for root, dirs, files in os.walk(saved_models_root):
        if any((f for f in files if f.endswith('.pt'))):

            fig, ax = plt.subplots(nrows=1, ncols=1)

            plot_model_logs_hist(root, chkpt_name='auto', ax=ax, show=False)

            # ax.set_title('/'.join(root.split('/')[-2:]))

            saving_path = os.path.join(params.metrics_root, 'loss_plot',
                                       root[len(saved_models_root)+1:] + '.png')

            plt.tight_layout()
            # plt.show()
            if not(os.path.isdir(os.path.dirname(saving_path))):
                os.makedirs(os.path.dirname(saving_path))
            plt.savefig(saving_path)


def plot_model_logs_hist(model_root, chkpt_name='auto', ax=None, show=True):

    if chkpt_name == 'auto':
        # max to get the last one in alphanumeric order
        chkpt_name = max(os.listdir(model_root))

    chkpt_path = os.path.join(model_root, chkpt_name)
    chkpt = torch.load(chkpt_path, map_location=DEVICE)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    plot_logs_hist(chkpt['logs'], ax=ax)

    if show:
        plt.show()


def plot_logs_hist(logs, ax=None):

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    # Extract info
    train_loss = torch.as_tensor(logs['_TrainingHistory__train_loss'])
    val_loss = torch.as_tensor(logs['_TrainingHistory__val_loss'])
    epoch_max = logs['_TrainingHistory__epoch']
    patience = logs['patience']
    margin = logs['margin']

    # Not plotting epoch 1, because training loss begins too high,
    # then shrinking the whole plot
    train_loss = train_loss[1:]
    val_loss = val_loss[1:]
    xaxis_offset = 2

    # Extrema
    mini = min(val_loss.min(), train_loss.min())
    maxi = max(val_loss.max(), train_loss.max())

    # Early stop boundaries
    es_xorg = epoch_max - patience
    es_yorg = (1+margin)*min(val_loss)
    ax.plot(np.array([es_xorg, epoch_max]),
            np.array([es_yorg, es_yorg]),
            '-.k')  # horizontal line
    ax.plot(np.array([es_xorg, es_xorg]),
            np.array([es_yorg, maxi]),
            '-.k', label='early stop window')  # vertical line

    # Train loss
    ax.plot(range(xaxis_offset, epoch_max+xaxis_offset-1), train_loss,
            color='tab:blue', linestyle='-', marker='.', label='training loss')

    # Val loss
    ax.plot(range(xaxis_offset, epoch_max+xaxis_offset-1), val_loss,
            color='tab:orange', linestyle='-', marker='.', label='validation loss')

    # Best model
    ax.plot(val_loss.argmin() + xaxis_offset, val_loss.min(), label='best', color='tab:green',
            linestyle='', marker='o', markersize=10, markeredgewidth=2, fillstyle='none')

    # Test is training finished
    val_loss_min = val_loss.min()
    val_loss_late_min = val_loss[-patience:].min()
    if not val_loss_late_min > ((1 + margin) * val_loss_min):
        ax.plot(val_loss[-patience:].argmin() + xaxis_offset + (epoch_max - patience - 1), val_loss_late_min,
                label='training not finished', color='tab:red',
                linestyle='', marker='o', markersize=7, markeredgewidth=0)

    # Custom xticks
    xticks = np.arange(0, epoch_max+1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:d}'.format(int(xt)) for xt in xticks])

    # Custom yticks
    yticks_step_wanted = 0.005  # x 100
    yticks_wanted = np.arange(
        mini*100, maxi*100+yticks_step_wanted, yticks_step_wanted)
    yticks = yticks_wanted / 100
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:3.3f}'.format(yt) for yt in yticks_wanted])
    ax.grid(True)

    # Labels etc.
    ax.set_xlabel('Epoch',
                  fontproperties=Font().axis_labels,
                  fontweight='bold')
    ax.set_ylabel('Mean MSE loss (x100)',
                  fontproperties=Font().axis_labels,
                  fontweight='bold')
    ax.legend()

    plt.tight_layout()


if __name__ == '__main__':
    main()
