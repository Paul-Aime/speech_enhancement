"""
As it is, the model input and output have the same shape.
There are 4 ways to proceed then :

#1 Keep it as it is, but putting a stride (=step) params in
the batchify function, putting it equals to nframes,
s.t. there is no overlapping

#2 Keep it as it is, adding the stride params to the
batchify function, but with stride < nframes, thus making
overlapping between windows, which has to be taen into
account for the reconstruction

#3 Only tak one frame (the middle one) form the ouput, may
be the easiest, but then : do we compute the loss only on
this frame or on the whole output, even if only taking the
middle frame for reconstruction

#4 Finally, we may want to compute the loss on y, not Y,
such that whatever the reconstruction method used, it is
taken into account, because then the general loss on the
reconstructed spectro is what matters. It may be the best.

Here we try to implement the option #4
"""
import torch

# TODO add an alignement param : 'left', 'right' or 'center'.
# Here only 'center' version is implemented


def main():

    # Params
    nfft = 10
    nframes = 3
    N = 3

    # STFT
    x = torch.randint(0, 10, (1, nfft, N))

    # Pad x such that X will have the same length
    n_padding_frames = int((nframes-1)/2)
    x_pad = pad(x, n_padding_frames)

    Y = batchify(x_pad, nframes)
    y = reconstruct(Y)

    # Print
    print(x.shape)
    print(x)
    print('\n\n')
    print(x_pad.shape)
    print(x_pad)
    print('\n\n')
    print(Y.shape)
    print(Y)
    print('\n\n')
    print(y.shape)
    print(y)


def reconstruct(X, apodisation=None):
    # TODO add apodisation, alignement and stride arguments

    # TODO
    # If apodisation is None, it means that we only keep one
    # frame for each output
    # Else, we need to do an overlapping summation

    # X of shape (B, C, H, W), with C=1
    # output of shape (C, H, B) in our case (i.e. stride=1, otherwise third dim < B, or idk yet)

    # nframes need to be odd, it's easier
    nframes = X.shape[3]

    # case for alignment == 'center' and apodisation == None
    idx_to_keep = int((nframes - 1) / 2)
    
    x = X[:, 0, :, idx_to_keep].T.unsqueeze(0)
    
    return x
    
    
    


def batchify(x, nframes):

    # nframes is the number of frames per input sample
    # c = x.shape[0] # Number of channels
    # h = x.shape[1]  # Height of the STFT, equals to nfft//2 +1
    w = x.shape[2]  # Width of the STFT, equals to the total number of frames

    X = torch.stack(tuple(x[0, :, i:i+nframes]
                          for i in range(w - nframes + 1)),
                    0).unsqueeze(dim=1)

    return X


def pad(x, nframes):
    # padding nframes to the left and the rigth, by simple replication
    # shape of x : (C, H, W)

    return torch.cat((x[:, :, :nframes], x, x[:, :, -nframes:]), dim=2)


if __name__ == "__main__":
    main()
