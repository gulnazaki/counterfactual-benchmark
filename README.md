# Benchmarking Counterfactual Image Generation
![image](https://github.com/gulnazaki/counterfactual-benchmark/assets/57211914/966b0d1f-3a3d-47c2-a77e-d32cf01d2868)

Code to reproduce our paper "Benchmarking Counterfactual Image Generation".

Published as a conference paper at [NeurIPS 2024 Datasets and Benchmarks Track](https://neurips.cc/virtual/2024/poster/97876).

You can find an interactive demo to compare counterfactual generation methods on our [project page](https://gulnazaki.github.io/counterfactual-benchmark/).

If you are interested, take a look at the [paper](https://arxiv.org/abs/2403.20287).

## Repository Organization
```
counterfactual_benchmark
├── datasets                                    # Code related to loading and transformations of datasets; We include MorphoMNIST, but CelebA has to be downloaded as described below.
│   ├── adni
│   │   └── preprocessing
│   │       ├── ...
│   │       └── README.md
│   ├── celeba
│   └── morphomnist
│       └── data
├── evaluation
│   ├── embeddings                              # Extraction of different embeddings
│   └── metrics                                 # Implementation of metrics
├── methods
│   └── deepscm                                 # Training and evaluation code for all methods under Deep-SCM paradigm
│       ├── checkpoints                         # Checkpoints (we only include predictors due to github size limitations)
│       │   ├── adni
│       │   │   └── trained_classifiers
│       │   ├── celeba
│       │   │   ├── complex
│       │   │   │   └── trained_classifiers
│       │   │   └── simple
│       │   │       └── trained_classifiers
│       │   └── morphomnist
│       │       └── trained_classifiers
│       ├── configs                             # Configuration files to reproduce experiments
│       │   ├── adni
│       │   ├── celeba
│       │   │   ├── complex
│       │   │   └── simple
│       │   ├── morphomnist
│       │   └── qualitative_grid.json
│       ├── evaluate.py                         # Evaluation script to test the trained models
│       ├── model.py                            # The SCM class that integrates the models for all mechanisms is defined here
│       ├── qualitative_grid.py                 # Script to produce a grid of qualitative results for all datasets and models
│       ├── train_classifier.py                 # Script to train the anti-causal predictors
│       ├── training_scripts.py
│       └── train.py                            # Script to train all models of the SCM
└── models                                      # Architectures and training/evaluation specifics for all types of models
    ├── classifiers
    ├── flows
    ├── gans
    └── vaes                                    # Both VAE and HVAE models are contained here
```

## Credits
We have based our repo on the structure provided by <https://github.com/rudolfwilliam/DeepBC>, from which we have also copied a large part of the Normalising Flow and VAE code. We have used the code provided by <https://github.com/biomedia-mira/causal-gen> for the HVAEs, VAEs and the predictors. The GAN models were adapted based on <https://github.com/wtaylor17/CDGMExplainers> and <https://github.com/vicmancr/CardiacAging> (for ADNI). For the processing of MorphoMNIST we have used code from <https://github.com/dccastro/Morpho-MNIST>, while for ADNI from <https://github.com/SANCHES-Pedro/adni_preprocessing>.

We want to thank all authors for their contributions and for providing their open-source code!

## Setup
```
virtualenv -p python3.10 venv
. venv/bin/activate
pip install -r requirements.txt
```

## How to Run
Inside `counterfactual_benchmark/methods/deepscm` the following can be used to train and evaluate any supported model for a given dataset.
We give an example for training and evaluating: (i) a VAE on MorphoMNIST (ii) a GAN on CelebA using the complex graph, (iii) a HVAE on ADNI. All experiments can be reproduced and extended with the configuration files inside `methods/deepscm/configs/`
```
python train.py -c configs/morphomnist/vae.json
python train_classifier.py -clf configs/morphomnist/classifier.json # this is optional as we provide classifier checkpoints
python evaluate.py -c configs/morphomnist/vae.json -clf configs/morphomnist/classifier.json
```

```
python train.py -c configs/celeba/complex/gan.json
python train_classifier.py -clf configs/celeba/complex/classifier.json
python evaluate.py -c configs/celeba/complex/gan.json -clf configs/celeba/complex/classifier.json
```

```
python train.py -c configs/celeba/adni/hvae.json
python train_classifier.py -clf configs/adni/classifier.json
python evaluate.py -c configs/adni/hvae.json -clf configs/adni/classifier.json
```

A description of all possible arguments for the evaluation script can be obtained with `python evaluate.py -h`


**To pretrain HVAE with the standard ELBO loss, the config should change to:**
```
"cf_fine_tune": False,
"evaluate_cf_model":False
```

**To fine-tune HVAE with the counterfactual loss described in the paper, the config should change to:**
```
"lr" : 1e-4 or 1e-5
...
"cf_fine_tune": True,
"classifiers_arch": "resnet" or "standard",
"ckpt_cls_path" : "<insert_classifiers_path>",

"evaluate_cf_model":False,
"elbo_constraint": "<set the elbo of the pretrained model>",
"checkpoint_path": "<insert_checkpoint_path>",
```

**To evaluate the fine-tuned HVAE the config should change to:**
```
"cf_fine_tune": True,
"evaluate_cf_model":True
```

**To fine-tune GAN with the cyclic cost minimisation described in the paper, the config should change to:**
```
"finetune": 1,
"pretrained_path": "<insert_checkpoint_path>",
```

## Datasets

![image](https://github.com/gulnazaki/counterfactual-benchmark/assets/57211914/8e8b5970-9474-4e06-b005-1a251341030b)

### Get CelebA Dataset
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


### Get ADNI Dataset
To request access to [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/) you should apply [here](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp).

After getting access (typically takes 1-2 days) you have to login in this [platform](https://ida.loni.usc.edu/login.jsp?project=ADNI).
Download the following:
- Download > Study Data > Study Info > Data & Database > ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]
- Download > Image Collections > Other Shared Collections > ADNI1:Complete 1Yr 1.5T (**all items and the csv**)

Move the downloaded csv files and the zip in `counterfactual_benchmark/datasets/adni/preprocessing/`. Then unzip and rename to `raw_data`.

Finally, follow the instructions in `counterfactual_benchmark/datasets/adni/preprocessing/README.md` to perform the preprocessing of ADNI

## Checkpoints for unconditional VAES (to compute CLD)
You can download from this google drive folder: https://drive.google.com/drive/folders/1SAak8-S3HOCmPyk8JFojAjV-xv2spSlY?usp=sharing

## How good is your counterfactual image generation?
![image](https://github.com/gulnazaki/counterfactual-benchmark/assets/57211914/ed125278-9c79-467d-9852-4693b319d91a)


## Citation
```
If you find this work helpful in your research, cite:
        @inproceedings{
          melistas2024benchmarking,
          title={Benchmarking Counterfactual Image Generation},
          author={Thomas Melistas and Nikos Spyrou and Nefeli Gkouti and Pedro Sanchez and Athanasios Vlontzos and Yannis Panagakis and Giorgos Papanastasiou and Sotirios A. Tsaftaris},
          booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
          year={2024},
          url={https://openreview.net/forum?id=0T8xRFrScB}
        }
```
