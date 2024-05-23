"""Components that were not implemented in the normflows package (at the time of writing this code), but needed here."""

from torch import nn
import torch.nn.functional as F
from normflows.flows import Flow
import torch
from torch.distributions import Categorical

class ConstAddScaleFlow(Flow):
    """
    Performs a simple affine transformation on the input with non-trainable parameters.
    """
    def __init__(self, const, scale):
        super().__init__()
        self.const = const
        self.scale = scale
      
    def forward(self, z):
        return z/self.scale - self.const, torch.ones(len(z), device=z.device)/self.scale
    
    def inverse(self, z):
        return (z + self.const)*self.scale, torch.ones(len(z), device=z.device)/self.scale
    
class SigmoidFlow(Flow):
    """
    Performs a sigmoid transformation on the input.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        log_deriv = (z - 2*torch.log(1 + torch.exp(z))).squeeze()
        return torch.sigmoid(z), log_deriv
    
    def inverse(self, z):
        inv = torch.log(z) - torch.log(1 - z)
        log_deriv = (z - 2*torch.log(1 + torch.exp(z))).squeeze()
        return inv, log_deriv


class SoftmaxCentered(Flow):
    """
    Implements softmax as a bijection, the forward transformation appends a value to the
    input and the inverse removes it. The appended coordinate represents a pivot, e.g., 
    softmax(x) = exp(x-c) / sum(exp(x-c)) where c is the implicit last coordinate.

    Adapted from a Tensorflow implementation: https://tinyurl.com/48vuh7yw 
    """
   # domain = constraints.real_vector
   # codomain = constraints.simplex

    def __init__(self, temperature: float = 1.):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        zero_pad = torch.zeros(*x.shape[:-1], 1, device=x.device)
        x_padded = torch.cat([x, zero_pad], dim=-1)
        return (x_padded / self.temperature).softmax(dim=-1)

    def inverse(self, y):
        log_y = torch.log(y.clamp(min=1e-12))
        unorm_log_probs = log_y[..., :-1] - log_y[..., -1:]
        return unorm_log_probs * self.temperature

    # def log_abs_det_jacobian(self, x: Tensor, y: Tensor): 
    #     """ -log|det(dx/dy)| """
    #     Kplus1 = torch.tensor(x.size(-1) + 1, dtype=x.dtype, device=x.device)
    #     return 0.5 * kp1.log() + torch.sum(x, dim=-1) - \
    #         Kplus1 * F.softplus(torch.logsumexp(x, dim=-1))

    def log_abs_det_jacobian(self, x, y):
        """ log|det(dy/dx)| """
        Kplus1 = torch.tensor(y.size(-1), dtype=y.dtype, device=y.device)
        return 0.5 * Kplus1.log() + torch.sum(torch.log(y.clamp(min=1e-12)), dim=-1)

    def forward_shape(self, shape: torch.Size):
        return shape[:-1] + (shape[-1] + 1,)  # forward appends one dim

    def inverse_shape(self, shape: torch.Size):
        if shape[-1] <= 1:
            raise ValueError
        return shape[:-1] + (shape[-1] - 1,)  # inverse removes last dim


class ArgMaxGumbelFlow(Flow):
    def __init__(self, logits):
        super().__init__()
        
        #self.context_nn = context_nn
        self.logits = logits
        self.categorical_distr = Categorical(logits=self.logits)
        
        

    def forward(self, u_gumbels):
       # logits = self.context_nn(context)
        y = u_gumbels + self.logits
        return y.argmax(dim=-1)
    
    # infer gumbel noise ε given observation x and parents pa_x
    def inverse(self, x):
       # logits = self.context_nn(context)
        
        uniforms = torch.rand(
            self.logits.shape,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )

        gumbels = -((-(uniforms.log())).log())

        mask = F.one_hot(
            x.squeeze(-1).to(torch.int64), num_classes=self.logits.shape[-1]
        )

        # (batch_size, 1) select topgumbel for truncation of other classes
        topgumbel = (mask * gumbels).sum(dim=-1, keepdim=True) - (mask * self.logits).sum(dim=-1, keepdim=True)

        mask = 1 - mask  # invert mask to select other != k classes
        g = gumbels + self.logits
        # (batch_size, num_classes)
        epsilons = -torch.log(mask * torch.exp(-g) + torch.exp(-topgumbel)) - (mask * self.logits)
      
        return epsilons
    
    def compute_log_prob(self, x):
        return -self.categorical_distr.log_prob(x.squeeze(-1)).unsqueeze(-1)
    

class GumbelConditionalFlow(Flow):
    def __init__(self, context_nn):
        super().__init__()
        self.context_nn = context_nn
    

    def condition(self, context):
        logits = self.context_nn(context)
       # print(logits)
        return ArgMaxGumbelFlow(logits=logits)
    

from torch.distributions.utils import _sum_rightmost

class GumbelCondFlow(Flow):

    def __init__(self, q0, flows, p=None): 
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p


    '''
    def forward_kld(self, x, context):
       
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            argmax_gumbel_flow = self.flows[i].condition(context)
            z = argmax_gumbel_flow.inverse(z)
            
    
            #z = self.flows[i].inverse(z, context)
            
       # print(self.q0.log_prob(z).shape, log_q.shape, z.shape)
        log_q += _sum_rightmost(self.q0.log_prob(z) , dim=1)
        
        #log_q += self.q0.log_prob(z[: , 0])

        return -torch.mean(log_q)
    
    '''
    
    def forward_kld(self, x, context):
        z = x
        log_prob = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            argmax_gumbel_flow = self.flows[i].condition(context)
            z = argmax_gumbel_flow.inverse(z)

            log_prob = log_prob - argmax_gumbel_flow.compute_log_prob(x)
            
    
        return -torch.mean(log_prob)

    
    def sample(self, context, num_samples=1):
        z = self.q0.sample((num_samples,))
        
        for flow in self.flows:
            argmax_gumbel_flow = flow.condition(context)
            z = argmax_gumbel_flow(z)
            
            #log_q -= log_det
        return z


class CondFlow(Flow):
    """
    Normalizing Flow model to approximate target distribution, with context layers.
    """
    def __init__(self, q0, flows, p=None): 
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z, context):
        for flow in self.flows:
            if flow.__class__.__name__ == 'MaskedAffineAutoregressive':
                z, _ = flow(z, context)
            else:
                z, _ = flow(z)
        return z

    def forward_and_log_det(self, z, context):
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            if flow.__class__.__name__ == 'MaskedAffineAutoregressive':
                z, log_d = flow(z, context)
            else:
                z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context):
        for i in range(len(self.flows) - 1, -1, -1):
            if self.flows[i].__class__.__name__ == 'MaskedAffineAutoregressive':
                x, _ = self.flows[i].inverse(x, context)
            else:
                x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x, context):
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if self.flows[i].__class__.__name__ == 'MaskedAffineAutoregressive':
                x, log_d = self.flows[i].inverse(x, context)
            else:
                x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x, context):
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            if self.flows[i].__class__.__name__  == 'MaskedAffineAutoregressive':
                z, log_det = self.flows[i].inverse(z, context)
            
            else:
                z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        raise NotImplementedError

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        raise NotImplementedError

    def sample(self, context, num_samples=1):
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            if flow.__class__.__name__ == 'MaskedAffineAutoregressive':
                z, log_det = flow(z, context)
            else:
                z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context):
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            if self.flows[i].__class__.__name__ == 'MaskedAffineAutoregressive':
                z, log_det = self.flows[i].inverse(z, context)
            else:
                z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


