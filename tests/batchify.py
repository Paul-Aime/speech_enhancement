import torch


def main():

    # Params
    nfft = 10
    nframes = 3
    N = 3

    # STFT
    x = torch.randint(0, 10, (1, nfft, N))

    # Pad x such that X will have the same length
    n_padding_frames = int((nframes-1)/2)
    x = pad(x, n_padding_frames)

    Y = batchify(x, nframes)

    # Print
    print(x.shape)
    print(x)
    print('\n\n')
    print(Y.shape)
    print(Y)


def pad(x, nframes):
    # padding nframes to the left and the rigth, by simple replication
    # shape of x : (C, H, W)

    return torch.cat((x[:, :, :nframes], x, x[:, :, -nframes:]), dim=2)


def batchify(x, nframes):

    # nframes is the number of frames per input sample
    # c = x.shape[0] # Number of channels
    # h = x.shape[1]  # Height of the STFT, equals to nfft//2 +1
    w = x.shape[2]  # Width of the STFT, equals to the total number of frames

    X = torch.stack(tuple(x[0, :, i:i+nframes]
                          for i in range(w - nframes + 1)),
                    0).unsqueeze(dim=1)

    return X


if __name__ == "__main__":
    main()
