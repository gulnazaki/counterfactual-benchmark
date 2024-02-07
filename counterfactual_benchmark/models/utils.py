"""Some functions that are needed here and there."""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from json import load
import argparse
import torch.nn as nn
from .weight_averaging import EMA

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def generate_checkpoint_callback(model_name, dir_path, monitor=None):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    filename= model_name + '-{epoch:02d}',
    monitor=monitor  # Disable monitoring for checkpoint saving
    )
    return checkpoint_callback

def generate_early_stopping_callback(patience=5):
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.0, patience=patience, mode = 'min')
    return early_stopping_callback

def generate_ema_callback(decay=0.999):
    ema_callback=EMA(decay=decay)
    return ema_callback

def flatten_list(list):
     return sum(list, [])

def get_config(config_dir, default):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default=default)
    args = argParser.parse_args()
    config = load(open(config_dir + args.name + ".json", "r"))
    return config

def override(f):
    return f

def overload(f):
    return f

def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f

def init_bias(m):
    if type(m) == nn.Conv2d:
        nn.init.zeros_(m.bias)
