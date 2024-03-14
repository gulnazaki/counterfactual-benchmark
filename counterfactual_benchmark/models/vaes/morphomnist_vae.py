from torch import nn
import torch
from collections import OrderedDict
from models.utils import flatten_list, init_bias
from models.vaes import CondVAE
from models.vaes.vae import DGaussNet


class Encoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, hidden_dim, n_chan=[1, 32, 32, 32], stride=[2, 2, 2],
                 kernel_size=[5, 3, 3], padding=[1, 1, 1]):
        super().__init__()
        self.n_chan = n_chan
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        activation_fn = nn.LeakyReLU()
        # conv layers
        self.conv = nn.Sequential(
            OrderedDict(flatten_list([
                [('enc' + str(i+1), nn.Conv2d(in_channels=n_chan[i], out_channels=n_chan[i+1],
                                              kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                ('enc' + str(i+1) + 'leaky_relu', activation_fn)] for i in range(len(n_chan) - 1)
            ]))
            )
        self.fc = nn.Sequential(nn.Linear(n_chan[-1] * 4 * 4, self.hidden_dim), activation_fn)
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
    def __init__(self, cond_dim, latent_dim, hidden_dim, n_chan=[32, 32, 32, 16], stride=[1, 1, 1],
                 kernel_size=[3, 3, 5], padding=[1, 1, 2]):
        super().__init__()
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("mu", torch.zeros(1, latent_dim))
        self.register_buffer("var", torch.ones(1, latent_dim))

        activation_fn = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim),
            activation_fn,
            nn.Linear(self.hidden_dim, self.n_chan[0] * 4 * 4),
            activation_fn
        )
        # decoder
        self.conv = torch.nn.Sequential(
            OrderedDict(flatten_list([[
                ('dec' + str(i+1) + 'upsample', nn.Upsample(scale_factor=2, mode="nearest")),
                ('dec' + str(i+1), nn.Conv2d(in_channels=self.n_chan[i], out_channels=self.n_chan[i+1],
                                                      kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                ('dec' + str(i+1) + 'relu', activation_fn)] for i in range(len(n_chan) - 1)
            ]))
        )

    def forward(self, u, cond):
        x = torch.cat([u, cond], dim=1) if self.cond_dim > 0 else u
        x = self.fc(x)
        x = x.view(-1, self.n_chan[0], 4, 4)
        x = self.conv(x)
        return x

    def prior(self, cond):
        return self.mu.repeat(cond.shape[0], 1), self.var.repeat(cond.shape[0], 1)


class MmnistCondVAE(CondVAE):
    def __init__(self, params, attr_size, name="image_vae", unconditional=False):
        # dimensionality of the conditional data
        cond_dim = sum(attr_size.values()) if not unconditional else 0
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
