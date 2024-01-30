"""This interface integrates all individual generative models into a single structural causal model (SCM). To be inherited from by individual projects."""

import torch
import os
import torch.nn as nn

class SCM(nn.Module):
    def __init__(self, ckpt_path, graph_structure, **models):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.graph_structure = graph_structure
        self.models = models
        self.__load_parameters()
        # no need for training further
        self.__freeze_models()
    
    def __load_parameters(self):
        # load pre-trained model for first file name starting with model name
        for name, model in self.models.items():   
            file_name = next((file for file in os.listdir(self.ckpt_path) if file.startswith(name)), None)
            model.load_state_dict(torch.load(self.ckpt_path + file_name, map_location=torch.device('cpu'))["state_dict"])
    
    def __freeze_models(self):
        for _, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False

    def encode(self, **xs):
        us = {}
        for var in self.graph_structure.keys():
            if len(self.graph_structure[var]) == 0:
                us[var] = self.models[var].encode(xs[var], torch.tensor([]).view(xs[var].shape[0], 0))
            else:
                us[var] = self.models[var].encode(xs[var], torch.cat([xs[pa] for pa in self.graph_structure[var]], dim=1))
        return us

    def decode(self, repl=None, **us):
        """If repl (replace) is not None, then the variables in repl (dict) are intervened on."""
        xs = {}
        for var in self.graph_structure.keys():
            if repl is None or var not in repl.keys():
                if len(self.graph_structure[var]) == 0:
                    xs[var] = self.models[var].decode(us[var], torch.tensor([]).view(us[var].shape[0], 0))
                else:
                    xs[var] = self.models[var].decode(us[var], torch.cat([xs[pa] for pa in self.graph_structure[var]], dim=1))
            else:
                # model intervention
                xs[var] = repl[var]
        return xs
    
    def decode_flat(self, us):
        """Required for backtracking algorithm. PyTorch does not support **kwargs for torch.autograd.functional.jacobian()"""
        xs = {}
        idx = 0
        for var in self.graph_structure.keys():
            if len(self.graph_structure[var]) == 0:
                xs[var] = self.models[var].decode(us[:, idx:(idx+self.models[var].latent_dim)], 
                                                  torch.tensor([]).view(us[:, idx:(idx+self.models[var].latent_dim)].shape[0], 0))
                idx += self.models[var].latent_dim
            else:
                xs[var] = self.models[var].decode(us[:, idx:(idx+self.models[var].latent_dim)], 
                                                  torch.cat([xs[pa] for pa in self.graph_structure[var]], dim=1))
                idx += self.models[var].latent_dim 
        return xs
    
    def sample(self, n_samp=1, std=1):
        """Sample new data points, conditional **xs."""
        us = {}
        for var in self.graph_structure.keys():
            # sample from prior
            samp = torch.normal(0, std, (n_samp, self.models[var].latent_dim))
            us[var] = samp
        xs = self.decode(**us)
        return xs, us