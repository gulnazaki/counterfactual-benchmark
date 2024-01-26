from scm.modules import CondVAE
from utils import flatten_list
from torch.functional import F
from torch import nn
import torch
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, n_chan=[1, 16, 32, 64, 128], stride=[1, 2, 2, 2], 
                 kernel_size=[4, 4, 4, 3], padding=[1, 1, 1, 0]):
        super().__init__()
        self.n_chan = n_chan
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        # conv layers
        self.conv = nn.Sequential(
            OrderedDict(flatten_list([
                [('enc' + str(i+1), nn.Conv2d(in_channels=n_chan[i], out_channels=n_chan[i+1], 
                                              kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                ('enc' + str(i+1) + 'relu', nn.ReLU())] for i in range(len(n_chan) - 1)
            ]))
            )
        self.fc = nn.Linear(n_chan[-1], self.latent_dim)
        # latent encoding
        self.mu = nn.Linear(self.latent_dim + self.cond_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim + self.cond_dim, self.latent_dim)

    def forward(self, x, cond):
        x = self.conv(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc(x)
        # get distribution components
        mu = self.mu(torch.cat([hidden, cond], dim=1))
        logvar = self.logvar(torch.cat([hidden, cond], dim=1))

        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, n_chan=[128, 64, 32, 16, 1], stride=[2, 2, 2, 2], 
                 kernel_size=[3, 4, 4, 4], padding=[0, 1, 0, 1]):
        super().__init__()
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.fc1 = nn.Linear(self.latent_dim + self.cond_dim, n_chan[0])
        # decoder
        self.conv1 = torch.nn.Sequential(
            OrderedDict(flatten_list([[
                ('dec' + str(i+1), nn.ConvTranspose2d(in_channels=n_chan[i], out_channels=n_chan[i+1], 
                                                      kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])),
                ('dec' + str(i+1) + 'relu', nn.ReLU())] for i in range(len(n_chan) - 2)
            ]))
        )
        # no relu for last layer 
        self.conv_fin = torch.nn.Sequential(
            OrderedDict([
                ('dec' + str(self.n_chan[-2]), nn.ConvTranspose2d(self.n_chan[-2], self.n_chan[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
            ])
        )

    def forward(self, u, cond):
        x = torch.cat([u, cond], dim=1)
        x = self.fc1(x)
        x = x.view(-1, self.n_chan[0], 1, 1)
        x = self.conv_fin(self.conv1(x))
        return x

class MmnistCondVAE(CondVAE):
    def __init__(self, cond_dim, latent_dim, name="image_vae", n_chan=[1, 16, 32, 64, 128], beta=4):
        # dimensionality of the conditional data
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        encoder = Encoder(cond_dim, latent_dim, n_chan=n_chan)
        decoder = Decoder(cond_dim, latent_dim, n_chan=n_chan[::-1])

        super(MmnistCondVAE, self).__init__(encoder, decoder, latent_dim, name=name)