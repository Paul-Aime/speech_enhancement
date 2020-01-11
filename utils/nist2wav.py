import os
import sys
from sphfile import SPHFile

# Adapted from https://stackoverflow.com/questions/44748258/reading-a-wav-file-from-timit-database-in-python


def main():

    assert len(sys.argv)==2, "root_dir needed as argument"
    root_dir = sys.argv[1]

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".WAV"):
                wav_file = os.path.join(root, file)
                nist2wav(wav_file)

    delete_WAV = input('Do you want to delete old .WAV ? y/[n]')

    if delete_WAV == 'y':
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".WAV"):
                    os.remove(os.path.join(root, file))


def nist2wav(wav_file):

    sph = SPHFile(wav_file)
    txt_file = ""
    txt_file = wav_file[:-3] + "TXT"

    f = open(txt_file,'r')
    for line in f:
        words = line.split(" ")
        start_time = (int(words[0])/16000)
        end_time = (int(words[1])/16000)
    print("writing file ", wav_file)
    sph.write_wav(wav_file.replace(".WAV",".wav"),start_time,end_time)


if __name__ == '__main__':
    main()