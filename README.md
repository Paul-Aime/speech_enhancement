# speech_enhancement

Implémentation inspirée de : ['A Fully Convolutional Neural Network for Speech Enhancement'.](https://paperswithcode.com/paper/a-fully-convolutional-neural-network-for)

## Prérequis

Installer conda et exécuter :

```bash
conda env create -f cpu-environment.yml
conda activate torch
```

ou :

```bash
conda env create -f gpu-environment.yml
conda activate torch-gpu
```
(tested with cuda 10.1)

ensuite on peut directement utiliser :

```
python train.py
```


## Exemple de résultat pour un RSB d'entrée de 0 dB

![Exemple de débruitage de STFT](docs/stfts_example.png?raw=true)



## Datasets

### Utilisé

Parole : [TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1)

### Autres

#### Parole

- [NTCD-TIMIT corpus](https://zenodo.org/record/1172064)
- [The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus / GitHub Dataset Description](https://github.com/philipperemy/timit)

#### Bruit

- [Babble Noise - Frequency-shaped babble noise generator](https://mynoise.net/NoiseMachines/babbleNoiseGenerator.php)
- [Freesound : babble noise](https://freesound.org/search/?q=babble)
- [Non speech noise (N1-N17 : crowd noise)](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html)