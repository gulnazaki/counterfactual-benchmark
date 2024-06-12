import sys
sys.path.append("../")
sys.path.append("../../")

from models.flows import GCondFlow
from models.flows.custom_components import CondFlow, GumbelCondFlow, GumbelConditionalFlow
import normflows as nf
from models.utils import override
from normflows.flows import affine
import torch
from json import load


class BaldFlow(GCondFlow):
    def __init__(self, params, name="Bald_flow", **kwargs):
        lr = params.get('lr', 1e-3)
        n_layers = params.get('n_layers', 1)
        super().__init__(name, lr, n_layers)
       # base = nf.distributions.base.DiagGaussian(1)
        base = torch.distributions.Gumbel(torch.zeros(1).cuda(), torch.ones(1).cuda())
        layers = []

        context_network = nf.nets.MLP([2, 64, 64, 2], init_zeros=True)  #context network

        for _ in range(n_layers):
            layers.append(GumbelConditionalFlow(context_nn=context_network))

        self.flow = GumbelCondFlow(base, layers)


class NoBeardFlow(GCondFlow):
    def __init__(self, params, name="No_Beard_flow", **kwargs):
        lr = params.get('lr', 1e-3)
        n_layers = params.get('n_layers', 1)
        super().__init__(name, lr, n_layers)
       # base = nf.distributions.base.DiagGaussian(1)
        base = torch.distributions.Gumbel(torch.zeros(1).cuda(), torch.ones(1).cuda())
        layers = []

        context_network = nf.nets.MLP([2, 64, 64, 2], init_zeros=True)  #context network

        for _ in range(n_layers):
            layers.append(GumbelConditionalFlow(context_nn=context_network))

        self.flow = GumbelCondFlow(base, layers)
