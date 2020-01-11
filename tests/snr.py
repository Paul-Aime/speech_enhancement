import torch

if True:  # Not to break code order with autoformatter
    # Needed here, and not under ifmain, because @time decorator is imported
    import sys
    from os import path
    sys.path.insert(1, path.dirname(path.dirname(path.abspath(__file__))))
    from utils.utils import time


ACTIVATED = True
NUMBER = 1000
PRINT_FUNC_CALL = False

"""Conclusion

Despite being a bit less readable, add_noise_snr1 is faster, and even
more as the number of samples grow
"""


def main():

    n_channels = 2
    n_samples = int(1e6)
    snr = 1

    sig = torch.randint(-10, 10, (n_channels, n_samples)).to(dtype=torch.float)
    noise = torch.randint(-10, 10, (n_channels, n_samples)
                          ).to(dtype=torch.float)

    # Check for consistency
    if not ACTIVATED:

        n_samples = int(5)

        y1 = add_noise_snr1(sig, noise, snr)
        y2 = add_noise_snr2(sig, noise, snr)

        print(y1)
        print(y2)

        print(torch.eq(y1, y2))

        assert torch.all(torch.eq(y1, y2))

    # Time it
    else:
        print('\n\n#1 add_noise_snr1:')
        print(add_noise_snr1(sig, noise, snr))

        print('\n\n#1 add_noise_snr2:')
        print(add_noise_snr2(sig, noise, snr))


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def add_noise_snr1(sig, noise, snr):
    """ shape [CxL] channel x length"""

    # Center sounds
    sig.add_(- sig.mean(dim=1).unsqueeze(0).T)
    noise.add_(- noise.mean(dim=1).unsqueeze(0).T)

    sig = sig.add(- sig.mean(dim=1).unsqueeze(0).T)
    noise = noise.add(- noise.mean(dim=1).unsqueeze(0).T)

    # Calcul addition coefficient
    alpha = ((torch.mean((sig**2), dim=1) /  # power of sig
              torch.mean(noise**2, dim=1))  # power of noise
             * (10 ** (-snr/10))).sqrt()

    return sig + (noise.T * alpha).T


@time(NUMBER, ACTIVATED, PRINT_FUNC_CALL)
def add_noise_snr2(sig, noise, snr):
    """ shape [CxL] channel x length"""

    # Transpose to ease computation lines of code
    sig.transpose_(dim0=1, dim1=0)
    noise.transpose_(dim0=1, dim1=0)

    # Center sounds
    sig.add_(- sig.mean(dim=0))
    noise.add_(- noise.mean(dim=0))

    # Calcul addition coefficient
    alpha = ((torch.mean((sig**2), dim=0) /  # power of sig
              torch.mean(noise**2, dim=0))  # power of noise
             * (10 ** (-snr/10))).sqrt()

    return (sig + (noise * alpha)).transpose_(dim0=1, dim1=0)


if __name__ == "__main__":
    main()
