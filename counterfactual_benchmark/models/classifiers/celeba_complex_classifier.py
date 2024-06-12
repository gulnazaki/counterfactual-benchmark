from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchvision import models
import torch.nn.functional as F
import sys
sys.path.append("../../")
from datasets.celeba.dataset import Celeba

class CelebaComplexClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 64, 64), n_chan = [3, 8, 16, 32, 32, 64, 1],
                 context_dim = 0, num_outputs = 1, lr=1e-3, version="standard"):
        super().__init__()
        self.variable = attr

        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()


        self.lr = lr
        self.variables =  {"Young": 0, "Male": 1, "No_Beard": 2, "Bald" : 3}
        self.conditions = {"Young":["No_Beard", "Bald"], "Male":["No_Beard", "Bald"],
                             "No_Beard":None, "Bald":None}

        #self.conditions = self.accociations[attr]
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]
        #n_chan = [3, 8, 16, 32, 32, 64, 1]

        self.context_dim = context_dim
        self.num_outputs = num_outputs

        if version == "standard":

            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=n_chan[0], out_channels=n_chan[1], kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=n_chan[1], out_channels=n_chan[2], kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=n_chan[2], out_channels=n_chan[3], kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_chan[3], out_channels=n_chan[4], kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.fc = nn.Sequential(
                nn.Linear(in_features=n_chan[4] + self.context_dim, out_features=n_chan[5]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=n_chan[5], out_features=n_chan[6]),
            )

        else:
            net = models.resnet18(pretrained=True)
            num_features = net.fc.in_features
            modules = list(net.children())[:-1]
            self.cnn = torch.nn.Sequential(*modules)


            self.fc = nn.Sequential(
                nn.Linear(in_features=num_features + self.context_dim, out_features=num_features),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=num_features, out_features=1),
            )





    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pooling

        if y is not None:
            x = torch.cat([x, y], dim=-1)

        return self.fc(x)


    def training_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train

        if self.variable == "Young" or self.variable == "Male":  #condition on No_Beard, Bald when training the classifier
            cond_no_beard = attrs_[:,self.variables["No_Beard"]].view(-1, 1)
            cond_bald = attrs_[:,self.variables["Bald"]].view(-1, 1)
            cond = torch.cat((cond_no_beard, cond_bald), dim=1)

            y_hat = self(x , cond)

        else:
            y_hat = self(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch):
        x, attrs_ = batch

        y = attrs_[:, self.attr] #select attribute to train

        if self.variable == "Young" or self.variable == "Male":  #condition on No_Beard, Bald when training the classifier
            cond_no_beard = attrs_[:,self.variables["No_Beard"]].view(-1, 1)
            cond_bald = attrs_[:,self.variables["Bald"]].view(-1, 1)
            cond = torch.cat((cond_no_beard, cond_bald), dim=1)

            y_hat = self(x , cond)

        else:
            y_hat = self(x)

        loss = nn.BCEWithLogitsLoss()(y_hat, y.type(torch.float32).view(-1, 1)) #applies sigmoid
        val_f1 = self.f1_score(y_hat, y.type(torch.long).view(-1,1))
        metrics = {'val_f1': val_f1, 'val_loss': loss,}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=[0.9, 0.999])
        return optimizer
