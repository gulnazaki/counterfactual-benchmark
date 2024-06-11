# Benchmarking Counterfactual Image Generation
![image](https://github.com/gulnazaki/counterfactual-benchmark/assets/57211914/966b0d1f-3a3d-47c2-a77e-d32cf01d2868)

Code to reproduce our paper "Benchmarking Counterfactual Image Generation".

[Pre-print](https://arxiv.org/abs/2403.20287)

## Repository Organization
```
counterfactual_benchmark
├── datasets                                    # Code related to loading and transformations of datasets; We include MorphoMNIST, but CelebA has to be downloaded as described below.
│   ├── celeba
│   └── morphomnist
│       └── data
├── evaluation
│   ├── embeddings                              # Extraction of different embeddings
│   └── metrics                                 # Implementation of metrics
├── methods
│   └── deepscm                                 # Training and evaluation code for all methods under Deep-SCM paradigm
│       ├── checkpoints
│       │   └── trained_classifiers             # MorphoMNIST predictors' checkpoints
│       ├── checkpoints_celeba
│       │   └── trained_classifiers             # CelebA predictors' checkpoints
│       ├── configs                             # Configuration files to reproduce experiments
│       ├── evaluate.py                         # Evaluation script to test the trained models
│       ├── model.py                            # The SCM class that integrates the models for all mechanisms is defined here
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
We give an example for training a VAE on MorphoMNIST. All experiments can be reproduced and extended with the configuration files inside `methods/deepscm/configs/`
```
python train.py -c configs/morphomnist_vae_config.json
python train_classifier.py -clf configs/morphomnist_classifier_config.json # this is optional as we provide classifier checkpoints
python evaluate.py -c configs/morphomnist_vae_config.json -clf configs/morphomnist_classifier_config.json
```
A description of all possible arguments for the evaluation script can be obtained with `python evaluate.py -h`


**To pretrain HVAE with the standard ELBO loss, the config should change to:**
```
"cf_fine_tune": False,
"evaluate_cf_model":False
```

**To fine-tune HVAE with the counterfactual loss described in the paper, the config should change to:**
```
"lr" : 1e-4,
...
"cf_fine_tune": True,
"classifiers_arch": "resnet" or "standard",
"ckpt_cls_path" : "<insert_classifiers_path>",

"evaluate_cf_model":False
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


## Get CelebA dataset
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

## How good is your counterfactual image?
![image](https://github.com/gulnazaki/counterfactual-benchmark/assets/57211914/ed125278-9c79-467d-9852-4693b319d91a)
