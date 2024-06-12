"implementation of HVAE class taken from https://github.com/biomedia-mira/causal-gen/blob/main/src/vae.py"
from typing import Any
from torch import nn
from torch.optim import AdamW
import pytorch_lightning as pl
import torch
import numpy as np
import random
import json
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier

import sys, os
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


class CondHVAE(pl.LightningModule):

    def __init__(self, encoder, decoder, likelihood, params, cf_fine_tune, evaluate, name):
        super().__init__()

        self.name = name
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.lr = params["lr"]
        self.weight_decay = params["wd"]
        self.beta = params["beta"]
        self.automatic_optimization = False
        self.evaluate = evaluate

        self.cond_prior =  json.loads(params["cond_prior"].lower())
        self.free_bits = params["kl_free_bits"]
        self.cf_fine_tune = cf_fine_tune

        self.lmbda = nn.Parameter(0.0 * torch.ones(1))
        self.elbo_constraint = params["elbo_constraint"]
        self.register_buffer("eps", self.elbo_constraint * torch.ones(1))


        if self.cf_fine_tune:
            if not self.evaluate:
                self.load_hvae_checkpoint_for_finetuning()   #load pretrained checkpoint

            device = "cuda"

            if self.params["context_dim"] == 2: #simple celeba graph
                
                self.anti_causal_cond = {
                                            "Smiling": [],
                                            "Eyeglasses": []
                                        }
                self.classifiers = {atr: CelebaClassifier(attr=atr, context_dim=len(list(self.anti_causal_cond[atr]))).eval()
                                        for atr in self.anti_causal_cond.keys()}



               # smiling_cls = CelebaClassifier(attr="Smiling").eval()
               # eye_cls = CelebaClassifier(attr="Eyeglasses").eval()
               # classifiers = {"Smiling":smiling_cls, "Eyeglasses": eye_cls}
               # file_name = next((file for file in os.listdir(self.params["ckpt_cls_path"]) if file.startswith(Smiling)), None)

               # smiling_cls.load_state_dict(torch.load(self.params["ckpt_cls_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
               # eye_cls.load_state_dict(torch.load(self.params["ckpt_cls_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
               # for key , c in classifiers.items():
               #     file_name = next((file for file in os.listdir(self.params["ckpt_cls_path"]) if file.startswith(key)), None)
               #     print(file_name)
               #     c.load_state_dict(torch.load(self.params["ckpt_cls_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])



              #  for model in [smiling_cls, eye_cls]:
              #      for param in model.parameters():
              #          param.requires_grad = False

              #  self.smiling_cls = smiling_cls.to(device)
              #  self.eye_cls = eye_cls.to(device)

            else: #complex celeba graph: context = 4
               # self.attributes = ["Young", "Male", "No_Beard", "Bald"]
                self.anti_causal_cond = {
                                            "Young": ["No_Beard", "Bald"],
                                            "Male": ["No_Beard", "Bald"],
                                            "No_Beard": [],
                                            "Bald": []
                                        }

                
                self.classifiers = {atr: CelebaComplexClassifier(attr=atr, context_dim=len(list(self.anti_causal_cond[atr])),
                                        version=self.params["classifiers_arch"]).eval() for atr in self.anti_causal_cond.keys()}


            for key , cls in self.classifiers.items():
                   # print(key)
                file_name = next((file for file in os.listdir(self.params["ckpt_cls_path"]) if file.startswith(key)), None)
                print(file_name)
                cls.load_state_dict(torch.load(self.params["ckpt_cls_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
                for param in cls.parameters():
                    param.requires_grad = False

                cls.to(device)


    def expand_parents(self, pa):
        return pa[..., None, None].repeat(1, 1, *(self.params["input_res"],) * 2) #expand the parents


    def load_hvae_checkpoint_for_finetuning(self):
        file_name = self.params["checkpoint_path"]
        print(file_name)
        device = "cuda"
        self.load_state_dict(torch.load(file_name, map_location=torch.device(device))["state_dict"])
        print("checkpoint loaded!")
        return

    def forward(self, x, parents, beta):
        acts = self.encoder(x)
        h, stats = self.decoder(parents=parents, x=acts)
        nll_pp = self.likelihood.nll(h, x)


        if self.free_bits > 0:
            free_bits = torch.tensor(self.free_bits).type_as(nll_pp)
            kl_pp = 0.0
            for stat in stats:
                kl_pp += torch.maximum(
                    free_bits, stat["kl"].sum(dim=(2, 3)).mean(dim=0)
                ).sum()
        else:
            kl_pp = torch.zeros_like(nll_pp)
            for _, stat in enumerate(stats):
                kl_pp += stat["kl"].sum(dim=(1, 2, 3))

        kl_pp = kl_pp / np.prod(x.shape[1:])  # per pixel
        kl_pp = kl_pp.mean()  # / self.log2
        nll_pp = nll_pp.mean()  # / self.log2

        nelbo = nll_pp + beta * kl_pp  # negative elbo (free energy)

        if torch.isnan(nelbo).sum() == 0 and kl_pp < 350:
            return dict(elbo=nelbo, nll=nll_pp, kl=kl_pp)

        else:
           # print("CHECK LOSSES FOR NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", dict(elbo=None, nll=None, kl=None))
            return dict(elbo=None, nll=None, kl=None)


    def sample(
        self, parents, return_loc = True, t= None):
        h, _ = self.decoder(parents=parents, t=t)
        return self.likelihood.sample(h, return_loc, t=t)



    def abduct(self, x, parents, cf_parents = None, alpha = 0.5, t = None):
        acts = self.encoder(x)
        _, q_stats = self.decoder(
            x=acts, parents=parents, abduct=True, t=t
        )  # q(z|x,pa)

        q_stats = [s["z"] for s in q_stats]

        if self.cond_prior and cf_parents is not None:
            _, p_stats = self.decoder(parents=cf_parents, abduct=True, t=t)  # p(z|pa*)
            p_stats = [s["z"] for s in p_stats]

            cf_zs = []
            t = torch.tensor(t).to(x.device)  # z* sampling temperature
            #print(q_stats)
            for i in range(len(q_stats)):
                # from z_i ~ q(z_i | z_{<i}, x, pa)
               # print(q_stats[i]["q_loc"])
                q_loc = q_stats[i]["q_loc"]
                q_scale = q_stats[i]["q_logscale"].exp()
                # abduct exogenouse noise u ~ N(0,I)
                u = (q_stats[i]["z"] - q_loc) / q_scale #U_zi
                # p(z_i | z_{<i}, pa*)
                p_loc = p_stats[i]["p_loc"]
                p_var = p_stats[i]["p_logscale"].exp().pow(2)

                # Option1: mixture distribution: r(z_i | z_{<i}, x, pa, pa*)
                #   = a*q(z_i | z_{<i}, x, pa) + (1-a)*p(z_i | z_{<i}, pa*)
                r_loc = alpha * q_loc + (1 - alpha) * p_loc
                r_var = (
                    alpha**2 * q_scale.pow(2) + (1 - alpha)**2 * p_var
                )  # assumes independence
                # r_var = a*(q_loc.pow(2) + q_var) + (1-a)*(p_loc.pow(2) + p_var) - r_loc.pow(2)

                # # Option 2: precision weighted distribution
                # q_prec = 1 / q_scale.pow(2)
                # p_prec = 1 / p_var
                # joint_prec = q_prec + p_prec
                # r_loc = (q_loc * q_prec + p_loc * p_prec) / joint_prec
                # r_var = 1 / joint_prec

                # sample: z_i* ~ r(z_i | z_{<i}, x, pa, pa*)
                r_scale = r_var.sqrt()
                r_scale = r_scale * t if t is not None else r_scale
                cf_zs.append(r_loc + r_scale * u)
            return cf_zs
        else:
            return q_stats  # zs


    def forward_latents(self, latents, parents, t = None, return_loc = False):
        h, _ = self.decoder(latents=latents, parents=parents, t=t)
        return self.likelihood.sample(h, return_loc=return_loc, t=t)



    def training_step(self, train_batch):
        x, cond = train_batch

        if self.cf_fine_tune:

            optimizer , lagrange_opt = self.optimizers()

            cond_ = self.expand_parents(cond)
            out = self.forward(x, cond_, beta = self.beta)
            nelbo_loss = out

            z , e , f_pa, obs = self.encode(x , cond)
            u = z , e , f_pa, obs
            cf_pa = cond

            if self.params["context_dim"] == 2: #simple causal graph
                attr = random.choice([0,1])

                if torch.rand(1) < 0.8: #select a parent to intervene
                    cf_pa[:,attr] = 1-cond[:,attr] #flip smile



                cf_x = self.decode(u, cf_pa)
                y_s_target = cf_pa[:, 0]
                y_e_target = cf_pa[:, 1]
                y_hat_s = self.classifiers["Smiling"](cf_x)
                y_hat_e = self.classifiers["Eyeglasses"](cf_x)
                smiling_cond_loss = nn.BCEWithLogitsLoss()(y_hat_s, y_s_target.type(torch.float32).view(-1, 1))
                eye_cond_loss = nn.BCEWithLogitsLoss()(y_hat_e, y_e_target.type(torch.float32).view(-1, 1))

                conditional_loss = 0.5*smiling_cond_loss + 0.5*eye_cond_loss

            else: #comple causal graph context_dim:4
                attr = random.choice([0,1,2,3])

                if torch.rand(1) < 0.8: #select a parent to intervene
                    cf_pa[:,attr] = 1-cond[:,attr] #flip attr


                cf_x = self.decode(u, cf_pa)
                y_young_target, y_male_target , y_no_beard_target, y_bald_target = cf_pa[:, 0], cf_pa[:, 1], cf_pa[:, 2], cf_pa[:, 3]

                condition = torch.cat((cf_pa[:,2].view(-1,1), cf_pa[:,3].view(-1,1)), dim = 1)

                y_hat_young = self.classifiers["Young"](cf_x, y = condition)
                y_hat_male = self.classifiers["Male"](cf_x, y = condition)
                y_hat_no_beard = self.classifiers["No_Beard"](cf_x)
                y_hat_bald = self.classifiers["Bald"](cf_x)

                young_cond_loss = nn.BCEWithLogitsLoss()(y_hat_young, y_young_target.type(torch.float32).view(-1, 1))
                male_cond_loss = nn.BCEWithLogitsLoss()(y_hat_male, y_male_target.type(torch.float32).view(-1, 1))
                no_beard_cond_loss = nn.BCEWithLogitsLoss()(y_hat_no_beard, y_no_beard_target.type(torch.float32).view(-1, 1))
                bald_cond_loss = nn.BCEWithLogitsLoss()(y_hat_bald, y_bald_target.type(torch.float32).view(-1, 1))

                conditional_loss = 0.25 * (young_cond_loss + male_cond_loss + no_beard_cond_loss + bald_cond_loss)



            optimizer.zero_grad(set_to_none=True)
            lagrange_opt.zero_grad(set_to_none=True)

            if conditional_loss!=None and out["elbo"]!=None:

                with torch.no_grad():
                    sg = self.eps - out["elbo"]

                damp = 100 * sg
                total_loss = conditional_loss - (self.lmbda - damp) * (self.eps - out["elbo"])
                self.manual_backward(total_loss)
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                self.lmbda.data.clamp_(min=0)


                self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

            else:
               total_loss = None


            return total_loss

        else:
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()

            cond_ = self.expand_parents(cond)
            out = self.forward(x, cond_, beta = self.beta)
            nelbo_loss = out

            optimizer.zero_grad()
            loss = nelbo_loss["elbo"]

            if loss!=None:

                self.manual_backward(loss)
                self.clip_gradients(optimizer, gradient_clip_val=350, gradient_clip_algorithm="norm")
                optimizer.step()

                for key , value in nelbo_loss.items():
                    self.log(key, value, on_step=False, on_epoch=True, prog_bar=True)

                if self.trainer.is_last_batch == 0:
                    scheduler.step()

                return loss

            return loss


    def validation_step(self, val_batch, batch_idx):
        x, cond = val_batch

        cond_ = self.expand_parents(cond)
        out = self.forward(x, cond_, beta=self.beta)

        nelbo_loss = out["elbo"]

        if self.cf_fine_tune: #

            cond_ = self.expand_parents(cond)
            out = self.forward(x, cond_, beta = self.beta)
            nelbo_loss = out

            z , e , f_pa, obs = self.encode(x , cond)
            u = z , e , f_pa, obs
            cf_pa = cond

            if self.params["context_dim"] == 2: #simple causal graph
                attr = random.choice([0,1])

                if torch.rand(1) < 0.8: #select a parent to intervene
                    cf_pa[:,attr] = 1-cond[:,attr] #flip smile



                cf_x = self.decode(u, cf_pa)
                y_s_target = cf_pa[:, 0]
                y_e_target = cf_pa[:, 1]
                y_hat_s = self.classifiers["Smiling"](cf_x)
                y_hat_e = self.classifiers["Eyeglasses"](cf_x)
                smiling_cond_loss = nn.BCEWithLogitsLoss()(y_hat_s, y_s_target.type(torch.float32).view(-1, 1))
                eye_cond_loss = nn.BCEWithLogitsLoss()(y_hat_e, y_e_target.type(torch.float32).view(-1, 1))

                conditional_loss = 0.5*smiling_cond_loss + 0.5*eye_cond_loss

            else: #comple causal graph context_dim:4
                attr = random.choice([0,1,2,3])

                if torch.rand(1) < 0.8: #select a parent to intervene
                    cf_pa[:,attr] = 1-cond[:,attr] #flip attr


                cf_x = self.decode(u, cf_pa)
                y_young_target, y_male_target , y_no_beard_target, y_bald_target = cf_pa[:, 0], cf_pa[:, 1], cf_pa[:, 2], cf_pa[:, 3]

                condition = torch.cat((cf_pa[:,2].view(-1,1), cf_pa[:,3].view(-1,1)), dim = 1)

                y_hat_young = self.classifiers["Young"](cf_x, y = condition)
                y_hat_male = self.classifiers["Male"](cf_x, y = condition)
                y_hat_no_beard = self.classifiers["No_Beard"](cf_x)
                y_hat_bald = self.classifiers["Bald"](cf_x)

                young_cond_loss = nn.BCEWithLogitsLoss()(y_hat_young, y_young_target.type(torch.float32).view(-1, 1))
                male_cond_loss = nn.BCEWithLogitsLoss()(y_hat_male, y_male_target.type(torch.float32).view(-1, 1))
                no_beard_cond_loss = nn.BCEWithLogitsLoss()(y_hat_no_beard, y_no_beard_target.type(torch.float32).view(-1, 1))
                bald_cond_loss = nn.BCEWithLogitsLoss()(y_hat_bald, y_bald_target.type(torch.float32).view(-1, 1))

                conditional_loss = 0.25 * (young_cond_loss + male_cond_loss + no_beard_cond_loss + bald_cond_loss)


            if conditional_loss!=None and out["elbo"]!=None:
                with torch.no_grad():
                    sg = self.eps - out["elbo"]

                damp = 100 * sg
                val_loss = conditional_loss - (self.lmbda - damp) * (self.eps - out["elbo"])


               # self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

            else:
               val_loss = torch.tensor(1.0).cuda()
            
            self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)


            return val_loss

        val_loss = nelbo_loss
        if nelbo_loss!=None:
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return nelbo_loss


    def configure_optimizers(self):

        if self.cf_fine_tune:
           # self.lr = 1e-5
            optimizer = AdamW([p for n, p in self.named_parameters() if n != "lmbda"], lr=self.lr,
                              weight_decay=self.weight_decay, betas=[0.9, 0.9])

            lagrange_opt = torch.optim.AdamW([self.lmbda], lr=1e-2, betas=[0.9, 0.9],
                                             weight_decay=0, maximize=True)


            return optimizer , lagrange_opt


        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.9])

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_warmup(100)
        )

        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}




    def encode(self, x, cond):
        cond =  self.expand_parents(cond)

        t = self.temperature if hasattr(self, 'temperature') else 0.1
        z = self.abduct(x, cond, t=t)

        if self.cond_prior:
            z = [z[i]['z'] for i in range(len(z))]

        rec_loc, rec_scale = self.forward_latents(z, parents=cond, return_loc=True)
        # abduct exogenous noise u

        eps = (x - rec_loc) / rec_scale.clamp(min=1e-12)

        return z, eps, cond, x


    def decode(self, u, cond, latent_mediator=False):
        z, e, pa, x = u

        # if using latent mediator we abduct also a counterfactual z
        if latent_mediator:
            t = self.temperature if hasattr(self, 'temperature') else 0.1
            z = self.abduct(x, pa, t=t, cf_parents=cond, alpha=0.65)

        t_u = 1  ##temp parameter

        cf_pa =  self.expand_parents(cond)

        cf_loc, cf_scale = self.forward_latents(z, parents=cf_pa, return_loc=True)


        cf_scale = cf_scale * t_u
        x = torch.clamp(cf_loc + cf_scale * e, min=-1, max=1)
        return x