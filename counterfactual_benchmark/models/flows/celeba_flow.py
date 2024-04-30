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
        lr = params.get('lr', 1e-6)
        n_layers = params.get('n_layers', 1)
        super().__init__(name, lr, n_layers)
       # base = nf.distributions.base.DiagGaussian(1)
        base = torch.distributions.Gumbel(torch.zeros(1), torch.ones(1))
        layers = []
       
        context_network = nf.nets.MLP([2, 64, 64, 2], init_zeros=True)  #context network
        
        for _ in range(n_layers):
            layers.append(GumbelConditionalFlow(context_nn=context_network))
        
        self.flow = GumbelCondFlow(base, layers)


class NoBeardFlow(GCondFlow):
    def __init__(self, params, name="No_Beard_flow", **kwargs):
        lr = params.get('lr', 1e-6)
        n_layers = params.get('n_layers', 1)
        super().__init__(name, lr, n_layers)
       # base = nf.distributions.base.DiagGaussian(1)
        base = torch.distributions.Gumbel(torch.zeros(1), torch.ones(1))
        layers = []
       
        context_network = nf.nets.MLP([2, 64, 64, 2], init_zeros=True)  #context network
        
        for _ in range(n_layers):
            layers.append(GumbelConditionalFlow(context_nn=context_network))
        
        self.flow = GumbelCondFlow(base, layers)



if __name__ == "__main__":
    
    
    attribute_size = {
        "Young": 1,
        "Male": 1,
        "No_Beard": 1,
        "Bald" : 1
    }


    config_file = "../../methods/deepscm/configs/celeba_complex_vae.json"
    with open(config_file, 'r') as f:
        config = load(f)

    params = config["mechanism_models"]["Bald"]["params"]

    context = torch.tensor([1., 1.]).unsqueeze(0)

    flow = BaldFlow(attribute_size=attribute_size, params=params)
    flow.load_state_dict(torch.load("../../methods/deepscm/checkpoints_celeba/trained_scm/Bald_flow-epoch=03.ckpt")["state_dict"])

    sample_bald = flow.flow.sample(context)
    print(sample_bald)