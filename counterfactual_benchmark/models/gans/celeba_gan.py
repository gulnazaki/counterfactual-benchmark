import os, sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict
import numpy as np
from collections import OrderedDict
from models.utils import flatten_list, continuous_feature_map
from models.gans import CondGAN


class Encoder(nn.Module):
    def __init__(self, latent_dim, num_continuous, n_chan=[3, 64, 128, 256, 256, 512, 512], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[2, 7, 5, 7, 4, 1], padding=[0, 0, 0, 0, 0, 0, 0]):
        super().__init__()

        self.num_continuous = num_continuous

        n_chan[0] = n_chan[0] + num_continuous
        self.n_chan = n_chan
        self.latent_dim = latent_dim

        activation_fn = nn.LeakyReLU(0.1)
        # conv layers
        self.layers = nn.Sequential(
            OrderedDict(flatten_list([
                [('enc' + str(i + 1), nn.Conv2d(in_channels=n_chan[i], out_channels=n_chan[i + 1],
                                                kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                 ('batchnorm'+str(i+1), nn.BatchNorm2d(n_chan[i + 1])),
                 ('enc' + str(i + 1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 2)
            ]))
        )

        lastconv = nn.Conv2d(in_channels=n_chan[-1], out_channels=latent_dim, kernel_size=kernel_size[-1], stride=stride[-1])
        batchnorm = nn.BatchNorm2d(latent_dim)
        self.layers.append(lastconv)
        self.layers.append(batchnorm)



    def forward(self, x: torch.Tensor, cond):


        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(x.shape[2], x.shape[3]))
        attr2 = continuous_feature_map(attr2, size=(x.shape[2], x.shape[3]))
        if cond.shape[1] <= 2:
            features = torch.concat((x, attr1, attr2), dim=1)
        else:
            attr3 = cond[:, 2]
            attr4 = cond[:, 3]
            attr3 = continuous_feature_map(attr3, size=(x.shape[2], x.shape[3]))
            attr4 = continuous_feature_map(attr4, size=(x.shape[2], x.shape[3]))
            features = torch.concat((x, attr1, attr2,attr3, attr4), dim=1)

        features = self.layers(features)

        return features


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_continuous, n_chan=[512 ,512, 256, 256, 128, 64, 3], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[4, 7, 5, 7, 2, 1], padding=[0, 0, 0, 0, 0, 0]):
        super().__init__()

        self.num_continuous = num_continuous
        self.latent_dim = latent_dim
        self.n_chan = n_chan
        n_chan[0] = n_chan[0] + num_continuous


        activation_fn = nn.LeakyReLU(0.1)
        self.layers = nn.Sequential(
            OrderedDict(flatten_list([
                [('gen' + str(i + 1), nn.ConvTranspose2d(in_channels=n_chan[i], out_channels=n_chan[i + 1],
                                                         kernel_size=kernel_size[i], stride=stride[i],
                                                         padding=padding[i])),
                 ('batchnorm'+str(i+1), nn.BatchNorm2d(n_chan[i + 1])),
                 ('gen' + str(i + 1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 2)
            ]))
        )
        lastconv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel_size[-1], stride=stride[-1], padding=padding[-1])
        batchnorm = nn.BatchNorm2d(3)
        sig = nn.Sigmoid()
        self.layers.append(lastconv)
        self.layers.append(batchnorm)
        self.layers.append(sig)

    def forward(self, u, cond):

        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(1, 1))
        attr2 = continuous_feature_map(attr2, size=(1, 1))
        if cond.shape[1] <= 2:
            features = torch.concat((u, attr1, attr2), dim=1)
        else:
            attr3 = cond[:, 2]
            attr4 = cond[:, 3]
            attr3 = continuous_feature_map(attr3, size=(u.shape[2], u.shape[3]))
            attr4 = continuous_feature_map(attr4, size=(u.shape[2], u.shape[3]))
            features = torch.concat((u, attr1, attr2,attr3, attr4), dim=1)

        features = self.layers(features)

        return features


class Discriminator(nn.Module):
    def __init__(self, num_continuous):
        super().__init__()

        self.num_continuous = num_continuous

        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1 + self.num_continuous+2, 64, (2, 2), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (7, 7), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, (5, 5), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, (7, 7), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 1024, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(2048, 2048, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(2048, 2048, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(2048, 1, (1, 1), (1, 1)),


        )


    def forward(self, x, u, cond):

        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(x.shape[2], x.shape[3]))
        attr2 = continuous_feature_map(attr2, size=(x.shape[2], x.shape[3]))
        if cond.shape[1] <= 2:
            features = torch.concat((x, attr1, attr2), dim=1)
        else:
            attr3 = cond[:, 2]
            attr4 = cond[:, 3]
            attr3 = continuous_feature_map(attr3, size=(x.shape[2], x.shape[3]))
            attr4 = continuous_feature_map(attr4, size=(x.shape[2], x.shape[3]))
            features = torch.concat((x, attr1, attr2,attr3, attr4), dim=1)


        dx = self.dx(features)
        dz = self.dz(u)
        z = self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))
        return z

class CelebaCondGAN(CondGAN):
    def __init__(self, params, attr_size, name="image_gan"):
        # dimensionality of the conditional data
        cond_dim = sum(attr_size.values())
        latent_dim = params["latent_dim"]
        num_continuous = params["num_continuous"]
        n_chan_enc = params["n_chan_enc"]
        n_chan_gen = params["n_chan_gen"]
        finetune= params["finetune"]
        lr = params["lr"]
        d_updates_per_g_update = params["d_updates_per_g_update"]
        gradient_clip_val = params["gradient_clip_val"]


        encoder = Encoder(latent_dim, num_continuous, n_chan=n_chan_enc)
        decoder = Decoder(latent_dim, num_continuous, n_chan=n_chan_gen)
        discriminator = Discriminator(num_continuous)

        super().__init__(encoder, decoder, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr, name)
