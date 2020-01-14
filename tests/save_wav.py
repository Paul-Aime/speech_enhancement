import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile


def main():
    fs = 8000
    signal = torch.load('./tests/SA1.signal')
    
    plt.figure()
    plt.plot(signal.squeeze().numpy())
    plt.show()
    
    wavfile.write('./tests/SA1_write.wav', fs, signal.squeeze().numpy())


if __name__ == "__main__":
    main()
