from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import torch.nn.functional as F
import sys
#from datasets.celeba.dataset import Celeba
sys.path.append("../../")
from datasets.celeba.dataset import Celeba

class CelebaComplexClassifier(pl.LightningModule):
    def __init__(self, attr, in_shape = (3, 64, 64), n_chan = [3, 8, 16, 32, 32, 64, 1], context_dim = 0, num_outputs = 1, lr=1e-3):
        super().__init__()
        self.variable = attr

        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        

        self.lr = lr
        self.variables =  {"Young": 0, "Male": 1, "No_Beard": 2, "Bald" : 3}
        self.attr = self.variables[attr] #select attribute
        in_channels = in_shape[0]
        #n_chan = [3, 8, 16, 32, 32, 64, 1]
       
        self.context_dim = context_dim
        self.num_outputs = num_outputs

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
        metrics = {'val_loss': loss, 'val_f1': val_f1}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)
       


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=[0.9, 0.999])
        return optimizer


#just test code for the classifiers
if __name__ == "__main__":

    attribute_size = {
        "Young": 1,
        "Male": 1,
        "No_Beard": 1,
        "Bald" : 1
    }

    train_set = Celeba(attribute_size, split="train")
  #  print(train_set[0])
  #  print(train_set[0][0].shape)
    tr_data_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    iterator = iter(tr_data_loader)
    batch = next(iterator)
    x , attrs = batch
    print(attrs, attrs.argmax(-1))
    print(x.shape)
   # print(attrs[: , 0].shape, attrs[:, 0].view(-1, 1).shape)
  #  print(attrs.shape, attrs[:,0], attrs[:, 1])
    cls_young = CelebaComplexClassifier(attr="Young", context_dim=2).eval()
    cls_male = CelebaComplexClassifier(attr="Male", context_dim=2).eval()
    cls_no_beard = CelebaComplexClassifier(attr="No_Beard").eval()
    cls_bald = CelebaComplexClassifier(attr="Bald").eval()

    condition = torch.cat((attrs[:,2].view(-1,1), attrs[:,3].view(-1,1)), dim = 1)
    print(condition.shape)
    out1= cls_young(x, y = condition)
    out2 = cls_male(x, y = condition)
    out3 = cls_no_beard(x)
    out4 = cls_bald(x)
    
    print("this is out1" , out1.shape)
    print("this is out2:", out2.shape)
    print("this is out3:", out3.shape)
    print("this is out4:", out4.shape)

    #print(attrs[:,0].view(-1,1).shape)