from torch import nn
import torch
import numpy as np
from collections import OrderedDict
from models.utils import init_bias
from models.vaes import CondVAE
from models.vaes.vae import DGaussNet


class Encoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, hidden_dim, n_chan=[1, 16, 24, 32, 64, 128, 256], stride=[1, 1, 1, 1, 1, 1],
                 kernel_size=[3, 3, 3, 3, 3, 3], padding=[1, 1, 1, 1, 1, 1], input_size=(1, 192, 192), num_convolutions=3):
        super().__init__()
        self.n_chan = n_chan
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        activation_fn = nn.LeakyReLU(0.1, inplace=True)

        # conv layers
        layers = OrderedDict()

        for i in range(len(n_chan) - 1):
            for j in range(0, num_convolutions):
                k = kernel_size[i] if j < num_convolutions - 1 else kernel_size[i] + 1
                s = stride[i] if j < num_convolutions - 1 else stride[i] + 1
                in_ch = n_chan[i] if j == 0 else n_chan[i + 1]

                layers[f'enc_{i+1}_conv_{j}'] = nn.Conv2d(in_channels=in_ch, out_channels=n_chan[i+1],
                                                          kernel_size=k, stride=s, padding=padding[i])
                layers[f'enc_{i+1}_batchnorm_{j}'] = nn.BatchNorm2d(n_chan[i+1])
                layers[f'enc_{i+1}_activation_fn_{j}'] = activation_fn

        self.conv = nn.Sequential(layers)

        self.intermediate_shape = np.array(input_size) // (2 ** (len(n_chan) - 1))
        self.intermediate_shape[0] = n_chan[-1]

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), self.hidden_dim), activation_fn
            )
        self.embed = nn.Sequential(nn.Linear(self.hidden_dim + self.cond_dim, self.hidden_dim), activation_fn)
        # latent encoding
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x, cond):
        batch, _, _, _ = x.shape
        x = self.conv(x).reshape(batch, -1)
        x = self.fc(x)
        hidden = self.embed(torch.cat((x, cond), dim=-1)) if self.cond_dim > 0 else self.embed(x)
        # get distribution components
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-9)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, hidden_dim, n_chan=[256, 128, 64, 32, 24, 16, 64], stride=[1, 1, 1, 1, 1, 1],
                 kernel_size=[3, 3, 3, 3, 3, 3], padding=[1, 1, 1, 1, 1, 1], output_size=(1, 192, 192), num_convolutions=3):
        super().__init__()
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("mu", torch.zeros(1, latent_dim))
        self.register_buffer("var", torch.ones(1, latent_dim))

        self.intermediate_shape = np.array(output_size) // (2 ** (len(n_chan) - 1))
        self.intermediate_shape[0] = n_chan[0]

        activation_fn = nn.LeakyReLU(0.1, inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            activation_fn,
            nn.Linear(self.hidden_dim, np.prod(self.intermediate_shape)),
            nn.BatchNorm1d(np.prod(self.intermediate_shape)),
            activation_fn
        )

        # decoder
        layers = OrderedDict()

        for i in range(len(n_chan) - 1):
            for j in range(0, num_convolutions):
                in_ch = n_chan[i] if j == 0 else n_chan[i + 1]

                if j < num_convolutions - 1:
                    layers[f'dec_{i+1}_conv_{j}'] = nn.Conv2d(in_channels=in_ch, out_channels=n_chan[i+1],
                                                              kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
                else:
                    layers[f'dec_{i+1}_conv_{j}'] = nn.ConvTranspose2d(in_channels=in_ch, out_channels=n_chan[i+1],
                                                                       kernel_size=kernel_size[i] + 1, stride=stride[i] + 1, padding=padding[i])
                layers[f'dec_{i+1}_batchnorm_{j}'] = nn.BatchNorm2d(n_chan[i+1])
                layers[f'dec_{i+1}_activation_fn_{j}'] = activation_fn

        self.conv = torch.nn.Sequential(layers)

    def forward(self, u, cond):
        x = torch.cat([u, cond], dim=1) if self.cond_dim > 0 else u
        x = self.fc(x)
        x = x.view(-1, *self.intermediate_shape)
        x = self.conv(x)
        return x

    def prior(self, cond):
        return self.mu.repeat(cond.shape[0], 1), self.var.repeat(cond.shape[0], 1)


class ADNICondVAE(CondVAE):
    def __init__(self, params, attr_size, name="image_vae", unconditional=False):
        # dimensionality of the conditional data
        cond_dim = params["context_dim"]
        latent_dim = params["latent_dim"]
        hidden_dim = params["hidden_dim"]
        n_chan = params["n_chan"]
        beta = params["beta"]
        lr = params["lr"]
        weight_decay = params["weight_decay"]
        fixed_logvar = params["fixed_logvar"]

        encoder = Encoder(cond_dim, latent_dim, hidden_dim, n_chan=n_chan)
        decoder = Decoder(cond_dim, latent_dim, hidden_dim, n_chan=n_chan[::-1][:-1] + [latent_dim])
        likelihood = DGaussNet(latent_dim, fixed_logvar, input_channels=1)

        super().__init__(encoder, decoder, likelihood, latent_dim, beta, lr, weight_decay, name)
        self.apply(init_bias)
