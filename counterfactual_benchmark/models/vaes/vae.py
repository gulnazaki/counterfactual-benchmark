"""Generic conditional VAE class without specified encoder and decoder archtitecture: to be implemented by subclasses."""

from scm.modules import StructuralEquation
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
import torch


class CondVAE(StructuralEquation, pl.LightningModule):
    def __init__(self, encoder, decoder, latent_dim, beta=4, lr=1e-6, name="image_vae"):
        super(CondVAE, self).__init__(latent_dim=latent_dim)
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.lr = lr

    def __reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        u = eps * std + mu
        return u

    def encode(self, x, cond, logvar=False):
        mu_u, logvar_u = self.encoder(x, cond)
        if logvar:
            return mu_u, logvar_u
        else:
            return mu_u

    def decode(self, u, cond):
        x = self.decoder(u, cond)
        return x

    def _vae_loss(self, xhat, x, mu, logvar, beta):
        mse_loss = nn.functional.mse_loss(xhat, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse_loss + beta * kl
        return loss

    def forward(self, x, cond):
        mu_u, logvar_u = self.encode(x, cond, logvar=True)
        u = self.__reparameterize(mu_u, logvar_u)
        x = self.decode(u, cond)
        return x, mu_u, logvar_u

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, cond = train_batch
        xhat, mu_u, logvar_u = self.forward(x, cond)
        loss = self._vae_loss(xhat, x, mu_u, logvar_u, beta=self.beta)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, train_batch, batch_idx):
        x, cond = train_batch
        xhat, mu_u, logvar_u = self.forward(x, cond)
        loss = self._vae_loss(xhat, x, mu_u, logvar_u, beta=self.beta)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
