import os
import torch
import matplotlib.pyplot as plt


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    folder = 'saved_models/park2017_R-CED9/fs8000_snr1_nfft256_hop128/'

    # max to get the last one in alphanumeric order
    chkpt_name = max(os.listdir(folder))
    chkpt_path = os.path.join(folder, chkpt_name)

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
