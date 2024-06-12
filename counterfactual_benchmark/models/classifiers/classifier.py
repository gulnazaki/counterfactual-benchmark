from torch.optim import Adam
#from germancredit.data.meta_data import attrs
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
import sys
sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike

class Classifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (1, 32, 32), width=8, num_outputs = 1, context_dim = 0 , lr=1e-3):
        super().__init__()
        self.variable = attr

        self.accuracy = Accuracy("multiclass", num_classes=10) #for the  digit

        self.lr = lr
        self.variables = {"thickness":0, "intensity":1, "digit": 2}
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]
        res = in_shape[1]
        s = 2 if res > 64 else 1
        activation = nn.LeakyReLU()

        '''CNN taken from https://github.com/biomedia-mira/causal-gen/blob/main/src/pgm/layers.py'''
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, s, 3, bias=False),
            nn.BatchNorm2d(width),
            activation,
            (nn.MaxPool2d(2, 2) if res > 32 else nn.Identity()),
            nn.Conv2d(width, 2 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 2 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 4 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 4 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 8 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8 * width),
            activation,
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * width + context_dim, 8 * width, bias=False),
            nn.BatchNorm1d(8 * width),
            activation,
            nn.Linear(8 * width, num_outputs),
        )

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pooling

        if y is not None:

            x = torch.cat([x, y], dim=-1)

        return self.fc(x)


    def training_step(self, batch, batch_idx):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train

        if self.variable == "digit":
            y = attrs_[:, self.attr:]

        if self.variable == "thickness":  #condition on intensity when training the thickness classifier
            cond = attrs_[:,self.variables["intensity"]].view(-1, 1)
            y_hat = self(x , cond)
            loss = nn.MSELoss()(y_hat, y.type(torch.float32).view(-1, 1))


        else:
            y_hat = self(x)

            if self.variable == "intensity":
                loss = nn.MSELoss()(y_hat, y.type(torch.float32).view(-1, 1))

            else: #digit
                loss = nn.CrossEntropyLoss()(y_hat, y.argmax(-1).type(torch.long))



        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        #loss = nn.CrossEntropyLoss()(y_hat, y.type(torch.long))
        return loss


    def validation_step(self, batch, batch_idx):
        x, attrs_ = batch
        y = attrs_[:, self.attr] #select attribute to train

        if self.variable == "digit":
            y = attrs_[:, self.attr:]


        if self.variable == "thickness":  #condition on intensity when training the thickness classifier
            cond = attrs_[:,self.variables["intensity"]].view(-1, 1)
            y_hat = self(x , cond)
            loss = nn.MSELoss()(y_hat, y.type(torch.float32).view(-1, 1))


        else:
            y_hat = self(x)

            if self.variable == "intensity":
                loss = nn.MSELoss()(y_hat, y.type(torch.float32).view(-1, 1))

            else: #digit
                loss = nn.CrossEntropyLoss()(y_hat, y.argmax(-1).type(torch.long))

                val_acc =   self.accuracy(y_hat, y.argmax(-1).type(torch.long))

                self.log('val_acc', val_acc, on_step=False , on_epoch=True, prog_bar=True)


        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
