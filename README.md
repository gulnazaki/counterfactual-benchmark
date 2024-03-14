# Benchmarking Counterfactual Image Generation
Code to reproduce our paper: "Benchmarking Counterfactual Image Generation".
Submitted to ECCV 2024.

## Repository Organization

- Code related to loading and transformations of datasets is included under `datasets`. We include the MorphoMNIST dataset, but CelebA has to be downloaded as described below.
- Code related to the extraction of embeddings and utilisation of metrics can be found in `evaluation`.

## Setup
```
virtualenv -p python3.10 venv
. venv/bin/activate
pip install -r requirements.txt
```

## How to Run


## CelebA
Images have to be downloaded from Google Drive and extracted:

See: https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM

```
mkdir celeba
sudo apt install unzip
mv celeba
unzip img_align_celeba.zip
```

Download attributes, etc. from:
- https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pblRyaVFSWGxPY0U&authuser=0
- https://drive.usercontent.google.com/download?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS&authuser=0
- https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pbThiMVRxWXZ4dU0&authuser=0
- https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ&authuser=0
- https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk&authuser=0

Move everything to `counterfactual_benchmark/datasets/celeba/data/`
