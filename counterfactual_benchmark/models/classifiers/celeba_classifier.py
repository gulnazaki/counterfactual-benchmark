from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import torch.nn.functional as F
import sys
sys.path.append("../../")
from datasets.celeba.dataset import Celeba

class CelebaClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 64, 64), num_outputs = 1, lr=1e-3, context_dim=0):
        super().__init__()
        self.variable = attr

        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()


        self.lr = lr
        self.variables = {"Smiling":0, "Eyeglasses":1}
        self.accociations = {"Smiling":None, "Eyeglasses":None}
        self.conditions = self.accociations[attr]
        #self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]

        self.num_outputs = num_outputs

        '''cnn layer implementation taken from https://openreview.net/forum?id=lZOUQQvwI3q'''
        self.cnn = nn.Sequential(
                        nn.Conv2d(3, 16, 3, 1, 1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, 3, 2, 1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 32, 3, 1, 1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, 2, 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, 1, 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, 2, 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                    )

        self.fc = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, self.num_outputs)
                )

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pooling

        return self.fc(x)


    def training_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train

        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr]

        y_hat = self(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid
        val_f1 = self.f1_score(y_hat, y.type(torch.long).view(-1,1))
        metrics = {'val_loss': loss, 'val_f1': val_f1}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=[0.9, 0.999])
        return optimizer
