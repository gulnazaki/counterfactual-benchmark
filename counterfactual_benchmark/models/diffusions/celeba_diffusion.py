from torch import nn
import torch
from collections import OrderedDict
from models.utils import flatten_list, init_bias
import sys
sys.path.append("../")
sys.path.append("../../")
from models.diffusions.diffusion import Diffusion


class CelebaConditionalDiffusion(Diffusion):
    def __init__(self, params, attr_size, name="image_diffusion"):
        # dimensionality of the conditional data
        cond_dim = sum(attr_size.values())
        sample_size = tuple(params["sample_size"])
        block_out_channels = tuple(params["block_out_channels"])
        cross_attention_dim = params["cross_attention_dim"]
        layers_per_block = int(params["layers_per_block"])
        attention_head_dim = int(params["attention_head_dim"])
        lr = params["lr"]
        self.name = name


        super().__init__(sample_size=sample_size, input_channels = 3, output_channels=3, 
                 block_out_channels = block_out_channels, cross_attention_dim = cross_attention_dim, layers_per_block=layers_per_block, 
                 attention_head_dim=attention_head_dim, cond_dim = cond_dim, lr=lr, name = self.name)
        self.apply(init_bias)