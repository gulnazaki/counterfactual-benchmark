"""Generic conditional flow class without specified archtitecture: to be implemented by subclasses."""

from torch.optim import Adam
import pytorch_lightning as pl

class GCondFlow(pl.LightningModule):
    def __init__(self, name, lr=1e-6, n_layers=3):
        super().__init__()
        self.name = name
        self.lr = lr
        self.n_layers = n_layers

    def forward(self, x, x_pa):
        return self.flow(x, x_pa)

    def encode(self, x, x_pa):
       # print(x, x_pa)
      #  print(self.flow)
        return self.flow.inverse(x, x_pa)

    def decode(self, u, x_pa):
        return self.flow(u, x_pa)

    def training_step(self, train_batch, batch_idx):
        x, x_pa = train_batch
        loss = self.flow.forward_kld(x, x_pa)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_pa = val_batch
        loss = self.flow.forward_kld(x, x_pa)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
