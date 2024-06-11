"""Generic conditional VAE class without specified encoder and decoder archtitecture: to be implemented by subclasses."""
from torch import nn
from torch.optim import Adam, AdamW
import pytorch_lightning as pl
import torch
import numpy as np

import sys
sys.path.append("../../")
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


class CondVAE(pl.LightningModule):
    def __init__(self, encoder, decoder, likelihood, latent_dim, beta=4, lr=1e-3, weight_decay=0.01, name="image_vae"):
        super().__init__()

        self.latent_dim = latent_dim
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
        mu_u, logvar_u = self.encoder(x, cond)
        u = sample_gaussian(mu_u, logvar_u)
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

        eps = (x - rec_loc) / rec_scale.clamp(min=1e-12)

        return  z , eps

    def decode(self, u, cond):
        z , e  = u
        t_u = self.temperature if hasattr(self, 'temperature') else 0.1
        cf_loc, cf_scale = self.forward_latents(z, parents=cond, return_loc=True)

        cf_scale = cf_scale * t_u
        x = torch.clamp(cf_loc + cf_scale * e, min=-1, max=1)
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.9]) if self.weight_decay > 0 else \
            Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(100)
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        # may happen if the last batch is of size 1 (dropout error)
        if train_batch[0].shape[0] == 1 and self.trainer.is_last_batch:
            return

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


class DGaussNet(nn.Module):
    def __init__(self, latent_dim, fixed_logvar, input_channels):
        super().__init__()
        self.x_loc = nn.Conv2d(
            latent_dim, input_channels, kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv2d(
            latent_dim, input_channels, kernel_size=1, stride=1
        )

        assert fixed_logvar == "False" or type(fixed_logvar) == float, \
            f'fixed_logvar can either be "False" or a float value, not: {fixed_logvar}'
        if fixed_logvar != "False":
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, fixed_logvar)
            self.x_logscale.weight.requires_grad = False
            self.x_logscale.bias.requires_grad = False

    def forward(self, h):
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=-9)
        return loc, logscale

    def approx_cdf(self, x):
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h, x):
        loc, logscale = self.forward(h)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(
        self, h, return_loc: bool = True, t=None):
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()
