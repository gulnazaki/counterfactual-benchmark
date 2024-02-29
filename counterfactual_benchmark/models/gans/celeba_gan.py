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
import math
import torch.nn.functional as F


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

        features = torch.concat((x, attr1, attr2), dim=1)
        
        features = self.layers(features)

        return features


class MappingNetwork(nn.Module):
    def __init__(self, latent_size = 512, hidden_size = 512, num_layers = 8):
        super(MappingNetwork, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #layers = [nn.Linear(latent_size, hidden_size)]
        #for _ in range(num_layers - 1):
        #    layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        #self.mapping = nn.Sequential(*layers)

        self.layers = []
        self.embs = []
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(latent_size, hidden_size).to('cuda'))
            self.embs.append(nn.Embedding(2,512).to('cuda'))
        self.layers.append(nn.Linear(latent_size, hidden_size).to('cuda'))

    def forward(self, x, cond):
        x = x.view(-1, self.latent_size).to('cuda')
        for i in range(self.num_layers - 2):
        
            x = self.layers[i](x).to('cuda')
            c = cond.to('cuda')
            c = c.matmul(self.embs[i].weight).reshape((-1, 512, 1, 1))
            c = c.squeeze()
            x = x+c
            
        x = self.layers[-1](x)
        return x
     #   x = x.view(-1, self.latent_size)
      #  return self.mapping(x)


class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features,
    ):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
    
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias    
class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise
class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_continuous, n_chan=[512 ,512, 256, 256, 128, 64, 3], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[4, 7, 5, 7, 2, 1], padding=[0, 0, 0, 0, 0, 0]):
        super().__init__()

        self.num_continuous = num_continuous
        self.latent_dim = latent_dim
        self.n_chan = n_chan
        n_chan[0] = n_chan[0] + num_continuous


        activation_fn = nn.LeakyReLU(0.1)
        self.map = MappingNetwork()
        
        self.starting_constant = nn.Parameter(torch.ones((1, 512, 64, 64)))
        self.initial_adain1 = AdaIN(512, 512)
        self.initial_adain2 = AdaIN(512, 512)
        self.initial_noise1 = InjectNoise(512)
        self.initial_noise2 = InjectNoise(512)
        self.initial_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        
        
        
        
        self.blocks=[]
        for i in range(9):
            self.blocks.append(GenBlock(512,512,512).to('cuda'))
                               
        self.rgb_layer = WSConv2d(in_channels = 512, out_channels = 3, kernel_size=1, stride=1, padding=0)
        
        self.layers = nn.Sequential(
            OrderedDict(flatten_list([
                [('gen' + str(i + 1), nn.ConvTranspose2d(in_channels=n_chan[i], out_channels=n_chan[i + 1],
                                                         kernel_size=kernel_size[i], stride=stride[i],
                                                         padding=padding[i])),
                 ('batchnorm'+str(i+1), nn.BatchNorm2d(n_chan[i + 1])),
                 ('gen' + str(i + 1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 2)
            ]))
        )
        lastconv = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=kernel_size[-1], stride=stride[-1], padding=padding[-1])
        batchnorm = nn.BatchNorm2d(3)
        self.sig = nn.Sigmoid()
        self.layers.append(lastconv)
        self.layers.append(batchnorm)
        self.layers.append(self.sig)

    def forward(self, u, cond):

        attr1 = cond[:, 0]
        attr2 = cond[:, 1]
        attr1 = continuous_feature_map(attr1, size=(1, 1))
        attr2 = continuous_feature_map(attr2, size=(1, 1))

        
        #w = self.map(noise)
        w = self.map(u, cond)
        x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)
        
        
        #upscaled = F.interpolate(out, scale_factor=2, mode="bilinear")
        for i in range(9):
            out = self.blocks[i](out, w)
        out = self.rgb_layer(out)
        out = self.sig(out)
    
        return out


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
        features = torch.concat((x, attr1, attr2), dim=1)

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
