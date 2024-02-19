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
# from datasets.morphomnist.dataset import MorphoMNISTLike
from models.utils import flatten_list, continuous_feature_map, init_weights
from models.gans import CondGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#Layer F K S BN D A
#Conv2D 64 (2,2) (1,1) Y 0.0 LReLU
#Conv2D 128 (7,7) (2,2) Y 0.0 LReLU
#Conv2D 256 (5,5) (2,2) Y 0.0 LReLU
#Conv2D 256 (7,7) (2,2) Y 0.0 LReLU
#Conv2D 512 (4,4) (1,1) Y 0.0 LReLU
#Conv2D 512 (1,1) (1,1) Y 0.0 Linea

class Encoder(nn.Module):
    def __init__(self, latent_dim, num_continuous, n_chan=[3, 64, 128, 256, 256, 512, 512], stride=[1, 2, 2, 2, 2, 2],
                 kernel_size=[2, 7, 5, 7, 4, 2], padding=[0, 0, 0, 0, 0, 0, 0]):
        super().__init__()

        self.num_continuous = num_continuous

        n_chan[0] = n_chan[0] + num_continuous
        self.n_chan = n_chan
        self.latent_dim = latent_dim

        activation_fn = nn.LeakyReLU(0.2)
        # conv layers
        self.layers = nn.Sequential(
            OrderedDict(flatten_list([
                [('enc' + str(i + 1), nn.Conv2d(in_channels=n_chan[i], out_channels=n_chan[i + 1],
                                                kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                 ('enc' + str(i + 1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 1)
            ]))
        )

        lastconv = nn.Conv2d(in_channels=n_chan[-1], out_channels=latent_dim, kernel_size=kernel_size[-1],
                             stride=stride[-1])
        self.layers.append(lastconv)


    def forward(self, x: torch.Tensor, cond):
        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(x.shape[2], x.shape[3]))
        attr2 = continuous_feature_map(attr2, size=(x.shape[2], x.shape[3]))
  
        features = torch.concat((x, attr1, attr2), dim=1)  
        features = self.layers(features)
        
        return features

#Layer F K S BN D A
#Conv2DT 512 (4,4) (1,1) Y 0.0 LReLU
#Conv2DT 256 (7,7) (2,2) Y 0.0 LReLU
#Conv2DT 256 (5,5) (2,2) Y 0.0 LReLU
#Conv2DT 128 (7,7) (2,2) Y 0.0 LReLU
#Conv2DT 64 (2,2) (1,1) Y 0.0 LReLU
#Conv2D 3 (1,1) (1,1) Y 0.0 Sigmoid
class Decoder(nn.Module):
    def __init__(self, latent_dim, num_continuous, n_chan=[512, 256, 256, 128, 64, 3], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[4, 7, 5, 7, 2, 1], padding=[0, 0, 0, 0, 0, 0]):
        super().__init__()

        self.num_continuous = num_continuous
        self.latent_dim = latent_dim
        self.n_chan = n_chan
        n_chan[0] = n_chan[0] + latent_dim + num_continuous


        activation_fn = nn.LeakyReLU(0.2)
        self.layers = nn.Sequential(
            OrderedDict(flatten_list([
                [('gen' + str(i + 1), nn.ConvTranspose2d(in_channels=n_chan[i], out_channels=n_chan[i + 1],
                                                         kernel_size=kernel_size[i], stride=stride[i],
                                                         padding=padding[i])),
                 ('gen' + str(i + 1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 1)
            ]))
        )
        lastconv = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=kernel_size[-1], stride=stride[-1])
        tahn = nn.Tanh()
        self.layers.append(lastconv)
        self.layers.append(tahn)

    def forward(self, u, cond):
      
        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(1, 1))
        attr2 = continuous_feature_map(attr2, size=(1, 1))
     
        features = torch.concat((u, attr1, attr2), dim=1)
    
      
        features = self.layers(features)
        
        return features


class Discriminator(nn.Module):
    def __init__(self, num_continuous):
        super().__init__()

        self.num_continuous = num_continuous
        self.digit_embedding = nn.Sequential(
            nn.Embedding(10, 256),
            nn.Unflatten(1, (1, 16, 16)),
            nn.Upsample(size=(32, 32)),
            nn.Tanh()
        )
        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1 + self.num_continuous + 1, 32, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 512, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, u, cond):
        processed_digit = self.digit_embedding(cond[:, 2:12].argmax(1))
        # processed_digit = self.digit_embedding(c["digit"].argmax(1))
        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1)
        attr2 = continuous_feature_map(attr2)
        # processed_continuous = {
        #     k: continuous_feature_map(v, size=(28, 28))
        #     for k, v in c.items()
        #     if k != "digit"
        # }
        features = torch.concat((x, processed_digit, attr1, attr2), dim=1)
        # features = torch.concat([X, processed_digit] + [
        #     processed_continuous[k]
        #     for k in sorted(processed_continuous.keys())], dim=1)
        dx = self.dx(features)
        dz = self.dz(u)
        z = self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))

        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


# enc chans [2, 64, 128, 256, 512]
# dec chans [ 256, 512, 256, 128, 64]
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

        super().__init__(encoder, decoder, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val,finetune, lr, name)


