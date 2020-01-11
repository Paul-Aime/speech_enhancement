import torch


def init_cuda(verbose=False):
    if torch.cuda.is_available():
        n_device = torch.cuda.device_count()
        device = torch.device('cuda:{:1d}'.format(n_device-1))
        if verbose:
            print('Using GPU : {}'.format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        if verbose:
            print('Using CPU')

    return device
