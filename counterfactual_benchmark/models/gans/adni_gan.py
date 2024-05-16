import sys
sys.path.append("../../")

import torch
import torch.nn as nn
from collections import OrderedDict
from models.utils import flatten_list, continuous_feature_map
from models.gans import CondGAN
from datasets.transforms import get_attribute_ids
from datasets.adni.dataset import bin_array, ordinal_array


class Encoder(nn.Module):
    def __init__(self, latent_dim, attr_size, n_chan=[1, 32, 64, 128, 256, 256, 512, 512], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[2, 7, 5, 7, 4, 1], padding=[0, 0, 0, 0, 0, 0, 0], input_res=192):
        super().__init__()

        self.attr_size_dict = attr_size
        self.attribute_ids = get_attribute_ids(self.attr_size_dict)

        self.attr_embedding = {
            attr_name: nn.Sequential(
                nn.Embedding(size, input_res**2 / 9),
                nn.Unflatten(1, (1, input_res / 3, input_res / 3)),
                nn.Upsample(size=(input_res, input_res)),
                nn.Sigmoid()
            )
            for attr_name, size in self.attr_size_dict.items() if size > 1
        }

        n_chan[0] = n_chan[0] + len(self.attr_size_dict)
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


    def attr_embedding_fn(self, attr_name, attr):
        if attr_name == "apoE":
            attr = bin_array(torch.round(attr), reverse=True)
        elif attr_name == "slice":
            attr = ordinal_array(torch.round(attr), reverse=True)
        else:
            raise RuntimeError(f"Non supported attribute: {attr_name}")
        return self.attr_embedding[attr_name](attr)


    def forward(self, x, cond):
        attrs = []
        for attr_name, size in self.attr_size_dict.items():
            attr = cond[:, self.attribute_ids[attr_name]]
            attr = continuous_feature_map(attr) if size == 1 else self.attr_embedding_fn(attr_name, attr)
            attrs.append(attr)

        features = torch.concat((x, *attrs), dim=1)
        features = self.layers(features)

        return features


class Decoder(nn.Module):
    def __init__(self, latent_dim, attr_size, n_chan=[512 ,512, 256, 256, 128, 64, 32, 1], stride=[1, 2, 2, 2, 1, 1],
                 kernel_size=[4, 7, 5, 7, 2, 1], padding=[0, 0, 0, 0, 0, 0], input_res=192):
        super().__init__()

        self.attr_size = attr_size
        self.attribute_ids = get_attribute_ids(self.attr_size_dict)

        self.attr_embedding = {
            attr_name: nn.Sequential(
                nn.Embedding(size, input_res**2 / 9),
                nn.Unflatten(1, (1, input_res / 3, input_res / 3)),
                nn.Upsample(size=(input_res, input_res)),
                nn.Sigmoid()
            )
            for attr_name, size in self.attr_size_dict.items() if size > 1
        }

        n_chan[0] = n_chan[0] + len(self.attr_size_dict)
        self.n_chan = n_chan
        self.latent_dim = latent_dim

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


    def attr_embedding_fn(self, attr_name, attr):
        if attr_name == "apoE":
            attr = bin_array(torch.round(attr), reverse=True)
        elif attr_name == "slice":
            attr = ordinal_array(torch.round(attr), reverse=True)
        else:
            raise RuntimeError(f"Non supported attribute: {attr_name}")
        return self.attr_embedding[attr_name](attr)


    def forward(self, u, cond):
        attrs = []
        for attr_name, size in self.attr_size_dict.items():
            attr = cond[:, self.attribute_ids[attr_name]]
            attr = continuous_feature_map(attr) if size == 1 else self.attr_embedding_fn(attr_name, attr)
            attrs.append(attr)

        features = torch.concat((u, *attrs), dim=1)

        features = self.layers(features)

        return features


class Discriminator(nn.Module):
    def __init__(self, attr_size, input_res=192):
        super().__init__()

        self.attr_size = attr_size
        self.attribute_ids = get_attribute_ids(self.attr_size_dict)

        self.attr_embedding = {
            attr_name: nn.Sequential(
                nn.Embedding(size, input_res**2 / 9),
                nn.Unflatten(1, (1, input_res / 3, input_res / 3)),
                nn.Upsample(size=(input_res, input_res)),
                nn.Sigmoid()
            )
            for attr_name, size in self.attr_size_dict.items() if size > 1
        }

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
            nn.Conv2d(1 + len(self.attr_size_dict), 64, (2, 2), (1, 1)),
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


    def attr_embedding_fn(self, attr_name, attr):
        if attr_name == "apoE":
            attr = bin_array(torch.round(attr), reverse=True)
        elif attr_name == "slice":
            attr = ordinal_array(torch.round(attr), reverse=True)
        else:
            raise RuntimeError(f"Non supported attribute: {attr_name}")
        return self.attr_embedding[attr_name](attr)


    def forward(self, x, u, cond):
        attrs = []
        for attr_name, size in self.attr_size_dict.items():
            attr = cond[:, self.attribute_ids[attr_name]]
            attr = continuous_feature_map(attr) if size == 1 else self.attr_embedding_fn(attr_name, attr)
            attrs.append(attr)

        features = torch.concat((x, *attrs), dim=1)

        dx = self.dx(features)
        dz = self.dz(u)
        z = self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))
        return z


class CelebaCondGAN(CondGAN):
    def __init__(self, params, attr_size, name="image_gan"):
        latent_dim = params["latent_dim"]
        n_chan_enc = params["n_chan_enc"]
        n_chan_gen = params["n_chan_gen"]
        finetune= params["finetune"]
        lr = params["lr"]
        d_updates_per_g_update = params["d_updates_per_g_update"]
        gradient_clip_val = params["gradient_clip_val"]

        encoder = Encoder(latent_dim, attr_size, n_chan=n_chan_enc)
        decoder = Decoder(latent_dim, attr_size, n_chan=n_chan_gen)
        discriminator = Discriminator(attr_size)

        super().__init__(encoder, decoder, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr, name)
