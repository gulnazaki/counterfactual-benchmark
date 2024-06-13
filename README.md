# Benchmarking Counterfactual Image Generation

Code to reproduce our paper "Benchmarking Counterfactual Image Generation".

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
python train.py -c configs/adni/hvae.json
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


## Get CelebA Dataset
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


## Get ADNI Dataset
To request access to [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/) you should apply [here](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp).

After getting access (typically takes 1-2 days) you have to login in this [platform](https://ida.loni.usc.edu/login.jsp?project=ADNI).
Download the following:
- Download > Study Data > Study Info > Data & Database > ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]
- Download > Image Collections > Other Shared Collections > ADNI1:Complete 1Yr 1.5T (**all items and the csv**)

Move the downloaded csv files and the zip in `counterfactual_benchmark/datasets/adni/preprocessing/`. Then unzip and rename to `raw_data`.

Finally, follow the instructions in `counterfactual_benchmark/datasets/adni/preprocessing/README.md` to perform the preprocessing of ADNI
