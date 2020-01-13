import os
import torch
import matplotlib.pyplot as plt


def main():

    # TODO Plot patience margin square

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    mode = 'auto' # Get the last trained model from model_dirpath folder

    saved_models_root = 'saved_models'
    experiment_folder = 'park2017_R-CED9'
    model_folder = 'fs8000_snr-20_nfft256_hop128'

    model_dirpath = os.path.join(saved_models_root, experiment_folder, model_folder)

    if mode == 'auto':
        # max to get the last one in alphanumeric order
        chkpt_name = max(os.listdir(model_dirpath))
    else:
        model_name = '016_0-001.pt'

    chkpt_path = os.path.join(model_dirpath, chkpt_name)

    chkpt = torch.load(chkpt_path, map_location=device)

    logs = chkpt['logs']
    epoch = logs['_TrainingHistory__epoch']
    train_loss = logs['_TrainingHistory__train_loss']
    val_loss = logs['_TrainingHistory__val_loss']

    print(chkpt_path)
    print(epoch)

    plt.figure()
    plt.plot(train_loss, '-b', label='train loss')
    plt.plot(val_loss, '-g', label='val loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
