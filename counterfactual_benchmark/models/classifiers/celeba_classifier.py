from torch.optim import Adam
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
import sys
sys.path.append("../../")
from datasets.celeba.dataset import Celeba

class CelebaClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 128, 128), width=64, num_outputs = 1, lr=1e-4):
        super().__init__()
        self.variable = attr

        self.accuracy = BinaryAccuracy()

        self.lr = lr
        # TODO add digit classifier
        self.variables = {"Smiling":0, "Eyeglasses":1}
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]
       # res = in_shape[1]
        #s = 2 if res > 64 else 1
        activation = nn.LeakyReLU()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            activation,
            #  (nn.MaxPool2d(2, 2) if res > 32 else nn.Identity()),
            nn.Conv2d(width, 2 * width, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 4 * width, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 4 * width, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 8 * width, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8 * width),
            activation,
            nn.Conv2d(8 * width, 8 * width, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8 * width),
            nn.Identity()   #activation,
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * width, 2 * width, bias=False),
            #nn.BatchNorm1d(8 * width),
            nn.ReLU(),
            nn.Linear(2 * width, 2 * width, bias=False),
            nn.ReLU(),
            nn.Linear(2 * width, num_outputs)
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
        val_acc =   self.accuracy(y_hat, y.type(torch.long).view(-1,1))

        self.log('val_acc', val_acc, on_step=False , on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


#just test code for the classifiers
if __name__ == "__main__":

    attribute_size = {
        "Smiling": 1,
        "Eyeglasses": 1
    }

    train_set = Celeba(attribute_size, split="train")
  #  print(train_set[0])
  #  print(train_set[0][0].shape)
    tr_data_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False)
    iterator = iter(tr_data_loader)
    batch = next(iterator)
    x , attrs = batch
    print(attrs, attrs.argmax(-1))
    print(x.shape)
    print(attrs[: , 0].shape, attrs[:, 0].view(-1, 1).shape)
  #  print(attrs.shape, attrs[:,0], attrs[:, 1])
    cls_smiling = CelebaClassifier(attr="Smiling", width=64).eval()
    cls_eyeglasses = CelebaClassifier(attr="Eyeglasses", width=64).eval()
    out1= cls_smiling(x)
    out2 = cls_eyeglasses(x)
  #  out3 = cls_digit(x)
    print("this is out1" , out1.shape)
    print("this is out2:", out2.shape)
    print(attrs[:,0].view(-1,1).shape)
  #  print(out2.shape)
  #  print(out3.shape)
  #  y = attrs[:, 2:]
  #  loss = F.cross_entropy(out3, y.argmax(-1).type(torch.long))
  #  print(loss)