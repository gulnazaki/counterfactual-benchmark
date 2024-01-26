"""Generic structural equation class without specified model class/archtitecture: Variational autoencoders or normalizing flows."""

from torch import nn

class StructuralEquation(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.latent_dim = latent_dim

    def encode(self):
        raise NotImplementedError

    def decode(self):
        raise NotImplementedError