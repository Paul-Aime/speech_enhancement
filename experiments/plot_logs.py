import os
import torch
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

    for root, dirs, files in os.walk(Params().saved_models_root):
        print('\n\n')
        print(' root:\n', root)
        print(' dirs:\n', dirs)
        print('files:\n', files)

        if any((f for f in files if f.endswith('.pt'))):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plot_model_logs_hist(root, chkpt_name='auto', ax=ax, show=False)
            ax.set_title('/'.join(root.split('/')[-2:]))
            plt.tight_layout()
            plt.show()
            
            # TODO save instead of show

        # experiment_folder = os.path.basename(os.path.dirname(model_root))
        # model_folder = os.path.basename(model_root)


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
    patience = logs['patience']
    margin = logs['margin']

    # Train loss
    ax.plot(train_loss, color='tab:blue', linestyle='-', label='training loss')

    # Val loss
    ax.plot(val_loss, color='tab:orange',
            linestyle='-', label='validation loss')

    # Best model
    ax.plot(val_loss.argmin(), val_loss.min(), label='best', color='tab:green',
            linestyle='', marker='o', markersize=10, markeredgewidth=2, fillstyle='none')

    # Early stop boundaries
    ax.plot([len(val_loss)-patience, len(val_loss)],
            [(1+margin)*min(val_loss), (1+margin)*min(val_loss)],
            '-.k')
    ax.plot([len(val_loss)-patience, len(val_loss)-patience],
            [(1+margin)*min(val_loss), max(max(val_loss), max(train_loss))],
            '-.k')

    # Labels etc.
    ax.set_xlabel('Epoch',
                  fontproperties=Font().axis_labels,
                  fontweight='bold')
    ax.set_ylabel('Mean MSE loss',
                  fontproperties=Font().axis_labels,
                  fontweight='bold')
    ax.legend()

    plt.tight_layout()


if __name__ == '__main__':
    main()
