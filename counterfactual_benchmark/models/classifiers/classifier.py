from torch.optim import Adam
from germancredit.data.meta_data import attrs
import torch.nn as nn
import pytorch_lightning as pl
import torch

class Classifier(pl.LightningModule):
    def __init__(self, attr, n_classes, n_hidden=10, lr=1e-3):
        super().__init__()
        self.attr = attr
        self.lr = lr
        self.ln = nn.Sequential(
            nn.Linear(in_features=len(attrs)-1, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_classes),
        )
    
    def forward(self, x):
        x = self.ln(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, attrs_ = batch
        y = attrs_[:, self.attr]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.type(torch.long))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, attrs_ = batch
        y = attrs_[:, self.attr]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.type(torch.long))
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer