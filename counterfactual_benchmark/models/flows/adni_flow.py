from models.flows import GCondFlow
from models.flows.custom_components import CondFlow, SigmoidFlow, ConstAddScaleFlow
import normflows as nf
from models.utils import override
from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive, AffineConstFlow
from normflows.flows import affine

class BrainVolFlow(GCondFlow):
    def __init__(self, params, name="brain_vol_flow", **kwargs):
        lr = params.get('lr', 0.005)
        n_layers = params.get('n_layers', 10)
        super().__init__(name, lr, n_layers)
        base = nf.distributions.base.DiagGaussian(1)
        layers = []
        layers.append(AffineConstFlow((1,)))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        # flow is conditional on "sex", "apoE", "age"
        layers.append(MaskedAffineAutoregressive(features=1, num_blocks=2, hidden_features=1, context_features=4))
        self.flow = CondFlow(base, layers)

class VentVolFlow(GCondFlow):
    def __init__(self, params, name="vent_vol_flow", **kwargs):
        lr = params.get('lr', 0.005)
        n_layers = params.get('n_layers', 10)
        super().__init__(name, lr, n_layers)
        base = nf.distributions.base.DiagGaussian(1)
        layers = []
        layers.append(AffineConstFlow((1,)))
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(1, 1, 1))
        # flow is conditional on "brain_vol", "age"
        layers.append(MaskedAffineAutoregressive(features=1, num_blocks=2, hidden_features=1, context_features=2))
        self.flow = CondFlow(base, layers)
