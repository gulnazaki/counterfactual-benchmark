import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DDIMInverseScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import logging

#logging.basicConfig(filename='train_diffusion.log', encoding='utf-8')

logging.basicConfig(filename='training.log', 
                    level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')


device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


from dataset import MorphoMNISTLike

class embedfc(nn.Module):
    def __init__(self, input_dim=12, emb_dim=128):
        super(embedfc, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ClassConditionedUnet(nn.Module):
  def __init__(self, context_dim=12, emb_dim=128):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
  #  self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=32,           # the target image resolution
        in_channels=1 + emb_dim, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),
        down_block_types=(
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

    self.fc_emb = embedfc(input_dim=context_dim, emb_dim=emb_dim)

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, y, drop_prob = 0.15):
    # Shape of x:
    bs, ch, w, h = x.shape
    # class conditioning in right shape to add as additional input channels
    #class_cond = self.class_emb(class_labels) # Map to embedding dimension
    #class_cond = self.fc_emb(c)
    mask = torch.rand(y.shape[0]) < drop_prob
    y_drop_out = y.clone()
    y_drop_out[mask] = 0
    
    context_cond = self.fc_emb(y_drop_out)
    context_cond = context_cond.view(bs, context_cond.shape[1], 1, 1).expand(bs, context_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, context_cond), 1) # (bs, 5, 28, 28)
    #print(net_input.shape)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)


    
attribute_size = {
        "thickness": 1,
        "intensity": 1,
        "digit": 10
    }

train_set = MorphoMNISTLike(attribute_size, train=True)
val_set =  MorphoMNISTLike(attribute_size, train=False)
print(len(train_set), len(val_set))

tr_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader  = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
print(len(tr_loader), len(val_loader))

model = ClassConditionedUnet(context_dim=12, emb_dim=128).to(device)


import numpy as np

n_epochs = 2000
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

loss_fn = nn.MSELoss()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []
val_losses = []
best_val_loss = np.inf

# The training loop
for epoch in range(n_epochs):
    losses = []
    val_losses = []
    for batch in tqdm(tr_loader):
        x , y = batch[0].to(device), batch[1].to(device)

        # Get some data and prepare the corrupted version
       # x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
       # y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = model(noisy_x, timesteps, y, drop_prob = 0.15) # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

   # print(f'Finished training epoch {epoch}. mean_train loss: {torch.tensor(losses).mean():05f}')
    logging.info(f'Finished training epoch {epoch}. mean_train loss: {torch.tensor(losses).mean():05f}')
   # print()


    for batch in tqdm(val_loader):

        x , y = batch[0].to(device), batch[1].to(device)

        # Get some data and prepare the corrupted version
       # x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
       # y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get some data and prepare the corrupted version
      #  x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
      #  y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)

        with torch.no_grad():
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
            pred = model(noisy_x, timesteps, y, drop_prob = 0) # Note that we pass in the labels y

        # Calculate the loss
            val_loss = loss_fn(pred, noise) # How close is the output to the noise
        
        val_losses.append(val_loss.item())
    current_val_loss = torch.tensor(val_losses).mean()
    
  #  print(f'Finished val epoch {epoch}. mean_val loss: {current_val_loss:05f}')
    logging.info(f'Finished val epoch {epoch}. mean_val loss: {current_val_loss:05f}' + "\n\n")
    #logging.info("\n\n")

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        print("Save checkpoint")
        torch.save(model.state_dict(), "checkpoint_morpho_2.pth")
    print()


