"""Generic conditional GAN class without specified encoder and decoder archtitecture: to be implemented by subclasses."""
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from models.structural_equation import StructuralEquation
import sys
import os
from torchmetrics.image.fid import FrechetInceptionDistance as FID

sys.path.append("../../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CondGAN(StructuralEquation, pl.LightningModule):
    def __init__(self, encoder, decoder, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr=1e-4, name="image_gan"):
        super(CondGAN, self).__init__(latent_dim=latent_dim)

        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.finetune = finetune
        self.lr = lr
        self.d_updates_per_g_update = d_updates_per_g_update
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False
        if self.finetune == 1:
            # Freeze the parameters of the decoder & discriminator
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False

    def encode(self, x, cond):
        return self.encoder(x, cond)

    def decode(self, u, cond):
        return self.decoder(u, cond)

    def discriminate(self, x, u, cond):
        return self.discriminator(x, u, cond)

    def gan_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y)
        return loss

    def mse_loss(self, x, xr):
        loss = torch.square(x - xr).mean()
        loss = torch.sqrt(loss)
        return loss


    def l1(self, z, ex):
        criterion = nn.L1Loss()
        loss = criterion(z,ex)
        return loss

    def forward_enc(self, x, cond):
        ex = self.encode(x, cond)
        return ex

    def forward_dec(self, u, cond):
        gu = self.decode(u, cond)
        return gu

    def forward_discr(self, x, z, cond):
        return self.discriminate(x, z, cond)

    def configure_optimizers(self):
        optimizer_E = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                       lr=self.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                       lr=self.lr, betas=(0.5, 0.999))
        return optimizer_E, optimizer_D


    def training_step(self, train_batch, batch_idx):
        if self.finetune == 1:
            optimizer_E, optimizer_D = self.optimizers()
            self.toggle_optimizer(optimizer_E)
            self.untoggle_optimizer(optimizer_D)
            x, cond = train_batch
            x = x.to(device)
            cond = cond.to(device)
            optimizer_E , _ = self.optimizers()
            optimizer_E.zero_grad()
            ex = self.forward_enc(x,cond)
            gu = self.forward_dec(ex, cond)
            loss = self.mse_loss(x, gu)
            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)
            latent = self.l1(z, ex)
            loss = loss + latent
            self.manual_backward(loss)
            optimizer_E.step()

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:

            x, cond = train_batch
            x = x.to(device)
            cond = cond.to(device)

            batch_size = x.shape[0]

            optimizer_E, optimizer_D = self.optimizers()
            valid = torch.ones((batch_size, 1), device=self.device)
            fake = torch.zeros((batch_size, 1), device=self.device)

            # sample noise
            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)

            ##########################
            # Optimize Discriminator #
            ##########################

            self.toggle_optimizer(optimizer_D)

            ex = self.forward_enc(x, cond)
            D_valid = self.forward_discr(x, ex, cond)
            loss_D_valid = self.gan_loss(D_valid, valid)

            gz = self.forward_dec(z, cond)
            D_fake = self.forward_discr(gz.detach(), z, cond)
            loss_D_fake = self.gan_loss(D_fake, fake)

            loss_D = loss_D_valid + loss_D_fake

            optimizer_D.zero_grad()
            self.manual_backward(loss_D)
            optimizer_D.step()
            self.untoggle_optimizer(optimizer_D)

            ##############################
            # Optimize Encoder & Decoder #
            ##############################

            # if batch_idx % self.d_updates_per_g_update == 0:
            self.toggle_optimizer(optimizer_E)

            EG_valid = self.forward_discr(gz.detach(), z, cond)
            loss_EG_valid = self.gan_loss(EG_valid, valid)

            EG_fake = self.forward_discr(x, ex, cond)
            loss_EG_fake = self.gan_loss(EG_fake, fake)

            loss_EG = loss_EG_valid + loss_EG_fake

            optimizer_E.zero_grad()
            self.manual_backward(loss_EG)
            #self.clip_gradients(optimizer_E, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer_E.step()
            self.untoggle_optimizer(optimizer_E)

            # if batch_idx % self.d_updates_per_g_update == 0:
            # loss_EG = loss_EG
            self.log({"eg_loss": loss_EG, "d_loss": loss_D}, on_step=False, on_epoch=True, prog_bar=True)
            return loss_EG


    def validation_step(self, train_batch):

        x, cond = train_batch
        x = x.to(device)
        cond = cond.to(device)

        with torch.no_grad():
            # sample noise
            z_mean = torch.zeros((len(x), self.latent_dim, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)
            gz = self.forward_dec(z, cond)

            ex = self.forward_enc(x, cond)
            gex = self.forward_dec(ex, cond)

            metric = FID(feature=64)
            metric.update(x, real=True)
            metric.update(gz, real=False)
            metric.update(gex, real=False)
            fid_score = metric.compute()

        self.log("fid", fid_score, on_step=False, on_epoch=True, prog_bar=True)

        epoch = self.current_epoch
        n_show = 10
        save_images_every = 10
        path = os.getcwd()
        image_output_path = path.replace('methods/deepscm', 'gantraining')
        if not os.path.exists(image_output_path):
            os.mkdir(image_output_path)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            with torch.no_grad():
                reals = []
                geners = []
                recons = []
                # generate images from same class as real ones
                for i in range(n_show):
                    images = x[i].to(device)
                    images = torch.unsqueeze(images, 0).float().to(device)


                    attrs = cond[i].to(device)
                    attrs = attrs.reshape((1, attrs.shape[0]))


                    z_mean = torch.zeros((len(images), 512, 1, 1)).float()
                    z = torch.normal(z_mean, z_mean + 1)
                    z = z.to(device)

                    gener = self.decode(z, attrs)
                    recon = self.decode(self.encode(images, attrs), attrs)
                    real = images.cpu().numpy()

                    gener = gener.reshape(gener.shape[1], gener.shape[2], gener.shape[3]).cpu().numpy()
                    recon = recon.reshape(recon.shape[1], recon.shape[2], recon.shape[3]).cpu().numpy()
                    real = real[0]


                    recons.append(recon)
                    geners.append(gener)
                    reals.append(real)

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                fig.subplots_adjust(wspace=0.05, hspace=0)
                plt.rcParams.update({'font.size': 20})
                fig.suptitle('Epoch {}'.format(epoch + 1))
                fig.text(0.04, 0.75, 'G(z, c)', ha='left')
                fig.text(0.04, 0.5, 'x', ha='left')
                fig.text(0.04, 0.25, 'G(E(x, c), c)', ha='left')



                if  geners[1].shape[0]==3:
                    recons.append(recon)
                    geners.append(gener)
                    reals.append(real)
                    for i in range(n_show):
                        geners[i] = np.transpose(geners[i], (1, 2, 0))
                        ax[0, i].imshow(geners[i])
                        ax[0, i].axis('off')
                        real = np.transpose(reals[i], (1, 2, 0))
                        ax[1, i].imshow(real)
                        ax[1, i].axis('off')
                        recons[i] = np.transpose(recons[i], (1, 2, 0))
                        ax[2, i].imshow(recons[i])
                        ax[2, i].axis('off')

                else:
                    recons.append(recon)
                    geners.append(gener)
                    reals.append(real)
                    for i in range(n_show):
                        ax[0, i].imshow(geners[i][0], cmap='gray', vmin=-1, vmax=1)
                        ax[0, i].axis('off')
                        ax[1, i].imshow(reals[i][0], cmap='gray', vmin=-1, vmax=1)
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recons[i][0], cmap='gray', vmin=-1, vmax=1)
                        ax[2, i].axis('off')

                plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png', format='png')
                plt.close()

        return