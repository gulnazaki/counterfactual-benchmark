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


    def abduct(self, x, parents):
        q_loc, q_logscale = self.encoder(x, cond=parents)  # q(z|x,pa)
        z = sample_gaussian(q_loc, q_logscale)
        return [z.detach()]


    def forward_latents(self, latents, parents, return_loc = False):
        h = self.decoder(cond=parents, u=latents[0])
        return self.likelihood.sample(h, return_loc)

    def encode(self, x, cond):
        z = self.abduct(x, cond)

        rec_loc, rec_scale = self.forward_latents(z, parents=cond, return_loc=True)
        # abduct exogenous noise u
        # t_u = 1
        # rec_scale = rec_scale * t_u
        eps = (x - rec_loc) / rec_scale.clamp(min=1e-12)

        return  z , eps


    def decode(self, u, cond):
        z , e  = u
        t_u = 0.1     #temp parameter
        cf_loc, cf_scale = self.forward_latents(z, parents=cond, return_loc=True)

        cf_scale = cf_scale * t_u
        x = torch.clamp(cf_loc + cf_scale * e, min=-1, max=1)
        return x

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
