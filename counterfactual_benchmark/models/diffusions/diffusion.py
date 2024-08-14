from torch import nn
from torch.optim import Adam, AdamW
import pytorch_lightning as pl
import torch
import numpy as np
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMInverseScheduler, DDIMScheduler
from tqdm import tqdm
import sys
sys.path.append("../../")
from models.utils import linear_warmup

class ClassConditionEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ClassConditionEncoder, self).__init__()
        layers = [
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).unsqueeze(1)
    

class Diffusion(pl.LightningModule):
    def __init__(self, sample_size=(64,64), input_channels = 3, output_channels=3, 
                 block_out_channels = (128, 256, 512, 512), cross_attention_dim = 512, layers_per_block=2, 
                 attention_head_dim=8, cond_dim = 2, lr=1e-4, name="image_diffusion"):
        super().__init__()
        self.lr = lr
        self.name = name
        
        self.model = UNet2DConditionModel(sample_size=sample_size, in_channels=input_channels, out_channels=output_channels,
                             block_out_channels = block_out_channels, 
                             cross_attention_dim=cross_attention_dim,  # Adjusted to match the embedding dimension
                             layers_per_block=layers_per_block,
                             attention_head_dim=attention_head_dim,
                             norm_num_groups=32)
        self.class_encoder  = ClassConditionEncoder(input_dim=cond_dim, embedding_dim=512)

        self.scheduler = DDPMScheduler() 
        
    

    def forward(self, x, t, y=None):
       # bs, _, w, h = x.shape
        class_embedding = self.class_encoder(y)

        return self.model(x, t, class_embedding).sample # (bs, 3, 64, 64)
    

#use diffusion model to conditionally sample from random noise
    def conditional_sample(self, cond):
        scheduler = DDIMScheduler() #DDPMScheduler() #DDIMScheduler()
        scheduler.set_timesteps(150)
        noisy_sample = torch.randn(1, 3, 64, 64).cuda()

        sample = noisy_sample
       # attrs_ = torch.tensor([[1., 0.]]).cuda()

        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
            with torch.no_grad():
                residual = self(sample, t, cond)

        # 2. compute less noisy image and set x_t -> x_t-1
            sample = scheduler.step(residual, t, sample).prev_sample

        return sample



    def encode(self, x, cond):

        inv_scheduler = DDIMInverseScheduler()
        inv_scheduler.set_timesteps(150)
        print(inv_scheduler.timesteps)
        current_img = x #img.unsqueeze(0).to(device)

       # inv_scheduler.set_timesteps(num_inference_steps=total_timesteps)
        timesteps = inv_scheduler.timesteps  #noise_scheduler.timesteps.flip(0)

        ## Encoding
        inv_scheduler.clip_sample = False
        condition = cond

        progress_bar = tqdm(timesteps)
        for i , t  in enumerate(progress_bar):  # go through the noising process
            with torch.no_grad():
                model_output = self(current_img, torch.Tensor((t,)).to(current_img.device), condition)
            current_img = inv_scheduler.step(model_output, t, current_img).prev_sample
            progress_bar.set_postfix({"timestep input": t})

        latent_img = current_img
        print(latent_img.shape)
        print(latent_img.min(), latent_img.max())
        z = latent_img
        return z 
       

    
    def decode(self, u, cond):
        noise_scheduler = DDIMScheduler()
        noise_scheduler.set_timesteps(150)
        sample = u
       # attrs_ = torch.tensor([[0., 0.]]).cuda() #condition #torch.tensor([[0, 1]]).cuda()

        for i, t in enumerate(tqdm.tqdm(noise_scheduler.timesteps)):
            with torch.no_grad():
                residual = self(sample, t, cond)

            sample = noise_scheduler.step(residual, t, sample).prev_sample

        x = sample
        return x
    

    def training_step(self, batch, batch_idx):
        x, attrs = batch

        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().cuda()
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)

        prediction = self(noisy_x, timesteps, attrs)
        
        loss = nn.MSELoss()(prediction, noise)
       
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    



    def validation_step(self, batch, batch_idx):
        x, attrs = batch

        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().cuda()
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)

        prediction = self(noisy_x, timesteps, attrs)
        
        loss = nn.MSELoss()(prediction, noise)

       

       
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer