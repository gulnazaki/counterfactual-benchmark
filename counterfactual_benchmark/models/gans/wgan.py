"""Generic conditional GAN class without specified encoder and decoder archtitecture: to be implemented by subclasses."""
from torch import nn
from torch.optim import Adam, AdamW
import torch.autograd as autograd
import pytorch_lightning as pl
import torch
import numpy as np
import sys
import os
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from models.utils import init_weights, rgbify

sys.path.append("../../")


def half_batch(batch_size, x, batch_idx):
    x = x[torch.randperm(batch_size)]
    even = batch_size % 2 == 0
    x1, x2 = x[:batch_size // 2], x[batch_size // 2 + (0 if even else 1):]
    return (x1, x2) if batch_idx % 2 == 0 else (x2, x1)

def calc_gradient_penalty(discriminator, real_data, fake_data, real_cond, fake_cond):
        'Compute grandient penalty for WGAN-GP (Arxiv:1704.00028)'
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(real_data).to(real_data.device)

        # Gradient w.r.t. 1st variable
        # Averaged images
        interpolation_im = epsilon * real_data + (1 - epsilon) * fake_data
        interpolation_im = autograd.Variable(
            interpolation_im, requires_grad=True).to(real_data.device)

        # TODO: Check the validity of using the target class index together
        # with the interpolated image for the gradient penalty loss.
        interpolation_logits = discriminator(interpolation_im, fake_cond)[0]
        grad_outputs = torch.ones(interpolation_logits.size(), device=real_data.device)
        gradients_a = autograd.grad(outputs=interpolation_logits,
                                    inputs=interpolation_im,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True)[0]

        gradients = gradients_a.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty


class CondWGAN(pl.LightningModule):
    def __init__(self, generator, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr=1e-4, name="image_gan"):
        super().__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.generator = generator
        self.discriminator = discriminator
        self.finetune = finetune
        self.lr = lr
        self.d_updates_per_g_update = d_updates_per_g_update
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False
        if self.finetune == 1:
            self.set_to_finetune()
        else:
            self.apply(init_weights)


    def encode(self, x, cond):
        return x, cond

    def decode(self, u, cond):
        x, prev_cond = u
        print(cond - prev_cond)
        return self.generator(x, cond - prev_cond)[0]

    def discriminate(self, x, u, cond):
        return self.discriminator(x, cond)

    def gan_loss(self, y_hat, y):
        return torch.mean(y.squeeze() * y_hat.squeeze())

    def l1_loss(self, z, ex):
        criterion = nn.L1Loss()
        loss = criterion(z, ex)
        return loss

    def l2_loss(self, x, xr):
        criterion = nn.MSELoss()
        loss = criterion(x, xr)
        return loss

    def configure_optimizers(self):
        if self.finetune == 0:
            optimizer_G = torch.optim.AdamW(self.generator.parameters(),
                                        lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            optimizer_D = torch.optim.AdamW(self.discriminator.parameters(),
                                        lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            return optimizer_G, optimizer_D
        else:
            optimizer_E = torch.optim.AdamW(list(self.encoder.parameters()),
                                        lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            return optimizer_E


    def set_to_finetune(self):
        # Freeze the parameters of the decoder & discriminator
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def training_step(self, train_batch, batch_idx):
        # may happen if the last batch is of size 1 (dropout error)
        if train_batch[0].shape[0] == 1 and self.trainer.is_last_batch:
            return

        x, cond = train_batch

        batch_size = x.shape[0]

        x_1, x_2 = half_batch(batch_size, x, batch_idx)
        cond_1, cond_2 = half_batch(batch_size, cond, batch_idx)

        optimizer_G, optimizer_D = self.optimizers()
        valid = -torch.ones((x_1.shape[0], 1), device=x.device)
        fake = torch.ones((x_1.shape[0], 1), device=x.device)

        x_gen, _map = self.generator(x_1, cond_2 - cond_1)

        # # sample noise
        # z_mean = torch.zeros((len(x), self.latent_dim)).float()
        # z = torch.normal(z_mean, z_mean + 1).to(x.device)
        # gz = self.forward_dec(z, cond)

        # ##########################
        # # Optimize Discriminator #
        # ##########################

        self.toggle_optimizer(optimizer_D)

        D_valid = self.discriminator(x_2, cond_2)
        loss_D_valid = self.gan_loss(D_valid, valid)

        D_fake = self.discriminator(x_gen.detach(), cond_2)
        loss_D_fake = self.gan_loss(D_fake, fake)

        # D_fake_2, D_fake_z = self.discriminator(x_1, cond_2)
        # loss_D_fake_2 = self.gan_loss(D_fake_2, fake)

        # D_fake_3, D_fake_z = self.discriminator(x_2, cond_1)
        # loss_D_fake_3 = self.gan_loss(D_fake_3, fake)

        gradient_penalty = calc_gradient_penalty(self.discriminator, x_2, x_gen.detach(), cond_1, cond_2)

        loss_D = loss_D_valid + loss_D_fake + 10 * gradient_penalty

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)


        ##############################
        # Optimize Encoder & Decoder #
        ##############################

        if batch_idx % self.d_updates_per_g_update == 0:
            self.toggle_optimizer(optimizer_G)

            G_valid = self.discriminator(x_gen, cond_2)
            loss_G_valid = self.gan_loss(G_valid, valid)

            # x_gen_2, _map, z_gen = self.generator(x_1, cond_1)
            # G_fake, G_z = self.discriminator(x_gen, cond_2)
            # loss_G_fake = self.gan_loss(G_fake, fake)
            # EG_fake = self.forward_discr(x, ex, cond)
            # loss_EG_fake = self.gan_loss(EG_fake, fake)

            # loss_EG = loss_EG_valid + loss_EG_fake

            # # reconstruct (cycle-consistency)
            # x_recon, _map, z_gen = self.generator(x_gen, cond_1 - cond_2)
            # cyc_loss = self.l1_loss(x_recon, x_1) * 1

            # Regularization loss
            # reg_loss = self.l1_loss(x_gen, x_1)

            # TODO try adding extra losses
            loss_G = loss_G_valid

            optimizer_G.zero_grad()
            self.manual_backward(loss_G)
            #self.clip_gradients(optimizer_E, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer_G.step()
            self.untoggle_optimizer(optimizer_G)
        else:
            loss_G = None

        if loss_G is not None:
            self.log_dict({"g_loss": loss_G, "d_loss": loss_D}, on_step=False, on_epoch=True, prog_bar=True)

        return loss_G


    def validation_step(self, train_batch, batch_idx):

        x, cond = train_batch

        batch_size = x.shape[0]

        x_1, x_2 = half_batch(batch_size, x, batch_idx)
        cond_1, cond_2 = half_batch(batch_size, cond, batch_idx)

        x_gen, _map = self.generator(x_1, cond_2 - cond_1)

        x_recon, _map = self.generator(x_gen, cond_1 - cond_2)

        metric = LPIPS(net_type='vgg', normalize=True).to(x.device)
        lpips_score = metric(rgbify(x_1), rgbify(x_recon))

        self.log("lpips", lpips_score, on_step=False, on_epoch=True, prog_bar=True)

            #     metric = FID(feature=64, normalize=True, reset_real_features=False).to(x.device)
            #     metric.update(rgbify(x), real=True)
            #     metric.update(rgbify(x_gen_random), real=False)
            #     metric.update(rgbify(x_gen), real=False)
            #     fid_score = metric.compute()

            # self.log("fid", fid_score, on_step=False, on_epoch=True, prog_bar=True)

        epoch = self.current_epoch
        n_show = 10
        save_images_every = 1
        path = os.getcwd()
        image_output_path = os.path.join(path, 'training_images_gan' + ('_finetuned' if self.finetune == 1 else ''))
        os.makedirs(image_output_path, exist_ok=True)

        if batch_idx == 0 and epoch % save_images_every == 0:
            reals = []
            geners = []
            recons = []
            # generate images from same class as real ones
            for i in range(n_show):
                images = x_1[i]
                images = torch.unsqueeze(images, 0).float()

                attrs_1 = cond_1[i]
                attrs_1 = torch.unsqueeze(attrs_1, 0)

                attrs_2 = cond_2[i]
                attrs_2 = torch.unsqueeze(attrs_2, 0)

                x_gen, _map = self.generator(images, attrs_2 - attrs_1)
                x_recon, _map = self.generator(x_gen, attrs_1 - attrs_2)
                gener = x_gen
                recon = x_recon
                real = images.cpu().numpy()

                gener = gener.squeeze().cpu().numpy()
                recon = recon.squeeze().cpu().numpy()
                real = real[0]


                recons.append(recon)
                geners.append(gener)
                reals.append(real)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.05, hspace=0)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('Epoch {}'.format(epoch))
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
                    ax[0, i].imshow(geners[i], cmap='gray', vmin=0, vmax=1)
                    ax[0, i].axis('off')
                    ax[1, i].imshow(reals[i][0], cmap='gray', vmin=0, vmax=1)
                    ax[1, i].axis('off')
                    ax[2, i].imshow(recons[i], cmap='gray', vmin=0, vmax=1)
                    ax[2, i].axis('off')

            plt.savefig(f'{image_output_path}/epoch-{epoch}.png', format='png')
            plt.close()

        return
