from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics.classification import F1Score, BinaryF1Score
import sys
from torchvision import models
sys.path.append("../../")
from models.classifiers.networks import CNN, MLP
from datasets.adni.dataset import bin_array, ordinal_array


class ADNIClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape=(1, 192, 192), num_outputs=1, 
                 children=None, num_slices=10, attribute_ids=None, lr=1e-4, arch="standard"):
        super().__init__()
        self.variable = attr

        if self.variable == "apoE":
            self.metric = lambda x, y: F1Score(task="multiclass", num_classes=3).to(x.device)\
                (bin_array(torch.round(x), reverse=True), bin_array(torch.round(y), reverse=True))
            self.loss = nn.CrossEntropyLoss()
        elif self.variable == "sex":
            self.metric = BinaryF1Score()
            self.loss = nn.BCEWithLogitsLoss()
        elif self.variable == "slice":
            self.metric = lambda x, y: F1Score(task="multiclass", num_classes=num_slices).to(x.device)\
                (ordinal_array(torch.round(x), reverse=True), ordinal_array(torch.round(y), reverse=True))
            self.loss = nn.CrossEntropyLoss()
        elif self.variable in ['age', 'brain_vol', 'vent_vol']:
            self.metric = nn.MSELoss()
            self.loss = nn.MSELoss()
        else:
            raise RuntimeError(f'Invalid attribute: {self.variable}')

        self.image_as_input = False
        if 'image' in children:
            self.image_as_input = True
            if arch == "resnet":
                net = models.resnet18(pretrained=True)
                net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                pretrained_weights = net.conv1.weight.clone()
                net.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)

                num_features = net.fc.in_features
                modules = list(net.children())[:-1]
                self.cnn = torch.nn.Sequential(*modules)

                self.fc = nn.Sequential(
                    nn.Linear(in_features=num_features + (len(children)-1), out_features=num_features),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=num_features, out_features=num_outputs),
                )

                self.network = torch.nn.Sequential(self.cnn, self.fc)
            else: 
                self.network = CNN(in_shape=in_shape, num_outputs=num_outputs, context_dim=len(children)-1)
            
            if 'image' in children:
                children.remove('image')
        else:
            self.network = MLP(num_inputs=len(children), num_outputs=num_outputs)
        self.cond_atts = children if children else None

        self.num_outputs = num_outputs
        self.attribute_ids = attribute_ids
        self.lr = lr


    def forward(self, x, y=None):

        if isinstance(self.network, MLP):
            return self.network.forward(x, y)

        else:
            x = self.network[0](x)
            x = x.mean(dim=(-2, -1))  # avg pooling

            if y is not None:
                x = torch.cat([x, y], dim=-1)

            return self.network[1](x)


    def training_step(self, batch, batch_idx):
        x, attrs_ = batch

        y = attrs_[:, self.attribute_ids[self.variable]]

        if self.cond_atts is not None:
            cond_indices = np.array([self.attribute_ids[att] for att in self.cond_atts]).flatten()
            cond = attrs_[:, cond_indices]
            y_hat = self(x, cond) if self.image_as_input else self(cond)
        else:
            y_hat = self(x)

        y_hat = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, attrs_ = batch

        y = attrs_[:, self.attribute_ids[self.variable]]

        if self.cond_atts is not None:
            cond_indices = np.array([self.attribute_ids[att] for att in self.cond_atts]).flatten()
            cond = attrs_[:, cond_indices]
            y_hat = self(x, cond) if self.image_as_input else self(cond)
        else:
            y_hat = self(x)

        y_hat = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y)

        val_metric = self.metric(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_metric', val_metric, on_step=False , on_epoch=True, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        return optimizer
