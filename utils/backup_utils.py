
import os
import re
import torch

# from utils.cuda_utils import init_cuda

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def load_model(model, optimizer, saved_model_path, verbose=False):

    if os.path.isfile(saved_model_path):
        saved_model = torch.load(saved_model_path, map_location=DEVICE)
        model.load_state_dict(saved_model['model_state_dict'])

        if verbose:
            print("Model state dict loaded ({}).\n".format(saved_model_path))

    else:
        raise FileNotFoundError(
            "Saved model not found at {}.".format(saved_model_path))

    if optimizer is not None:
        if 'optimizer_state_dict' in saved_model:
            optimizer.load_state_dict(saved_model['optimizer_state_dict'])

    return saved_model['logs']


def save_checkpoint(model, optimizer, loss, logs, params):

    model_saving_path = get_model_saving_path(logs.epoch, loss, params)

    dirname = os.path.dirname(model_saving_path)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    else:
        # Empty the dir if a chkpt with equal epoch already exist
        chkpts = os.listdir(dirname)
        chkpts_epochs = [int(re.match(r'(\d+)_', chkpt).group(1))
                         for chkpt in chkpts]
        if logs.epoch in chkpts_epochs:
            for c in [os.path.join(dirname, chkpt) for chkpt in chkpts]:
                os.remove(c)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logs': logs.__dict__,
    }, model_saving_path)

    return model_saving_path


def get_model_saving_path(epoch, loss, params):

    loss_str = "".join([l if l != '.' else '-'
                        for l in "{:.3f}".format(loss)])

    model_saving_path = os.path.join(params.model_saving_dir,
                                     '{:03d}_{}.pt'.format(epoch,
                                                           loss_str))

    return model_saving_path
