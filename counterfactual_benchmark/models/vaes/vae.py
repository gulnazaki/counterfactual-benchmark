"""Generic conditional VAE class without specified encoder and decoder archtitecture: to be implemented by subclasses."""
from torch import nn
from torch.optim import AdamW
import pytorch_lightning as pl
import torch
import numpy as np

import sys
sys.path.append("../../")
from models.structural_equation import StructuralEquation
from models.utils import linear_warmup

@torch.jit.script
def sample_gaussian(loc, logscale):
    return loc + logscale.exp() * torch.randn_like(loc)

@torch.jit.script
def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )


class CondVAE(StructuralEquation, pl.LightningModule):
    def __init__(self, encoder, decoder, likelihood, latent_dim, beta=4, lr=1e-3, weight_decay=0.01, name="image_vae"):
        super(CondVAE, self).__init__(latent_dim=latent_dim)
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay

    def encode(self, x, cond, logvar=False):
        mu_u, logvar_u = self.encoder(x, cond)
        if logvar:
            return mu_u, logvar_u
        else:
            return mu_u

    def decode(self, u, cond):
        h = self.decoder(u, cond)
        x = self.likelihood.sample(h)
        return x

    def _vae_loss(self, h, x, mu, logvar, prior_mu, prior_var, beta):
        nll_pp = self.likelihood.nll(h, x)
        kl_pp = gaussian_kl(mu, logvar, prior_mu, prior_var)
        kl_pp = kl_pp.sum(dim=-1) / np.prod(x.shape[1:])
        loss = nll_pp.mean() + beta * kl_pp.mean()
        return loss

    def forward(self, x, cond):
        mu_u, logvar_u = self.encode(x, cond, logvar=True)
        u = sample_gaussian(mu_u, logvar_u)
        # u = self.__reparameterize(mu_u, logvar_u)
        h = self.decoder(u, cond)
        return h, mu_u, logvar_u

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.9])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(100)
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        x, cond = train_batch
        h, mu_u, logvar_u = self.forward(x, cond)
        prior_mu, prior_var = self.decoder.prior(cond)
        loss = self._vae_loss(h, x, mu_u, logvar_u, prior_mu, prior_var, beta=self.beta)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, train_batch, batch_idx):
        x, cond = train_batch
        h, mu_u, logvar_u = self.forward(x, cond)
        prior_mu, prior_var = self.decoder.prior(cond)
        loss = self._vae_loss(h, x, mu_u, logvar_u, prior_mu, prior_var, beta=self.beta)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
