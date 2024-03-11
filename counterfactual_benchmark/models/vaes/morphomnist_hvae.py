"HVAE implementation taken from https://github.com/biomedia-mira/causal-gen/blob/main/src/vae.py"
from typing import Dict, List, Optional, Tuple

import sys
import json
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.functional import F
from collections import OrderedDict
from models.utils import init_bias
from models.vaes import CondHVAE
from json import load

import sys
#sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike

#from hps import Hparams

EPS = -9  # minimum logscale


@torch.jit.script
def gaussian_kl(q_loc: Tensor, q_logscale: Tensor, p_loc: Tensor, p_logscale: Tensor) -> Tensor:
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )


@torch.jit.script
def sample_gaussian(loc: Tensor, logscale: Tensor) -> Tensor:
    return loc + logscale.exp() * torch.randn_like(loc)


class Block(nn.Module):
    def __init__(
        self,
        in_width: int,
        bottleneck: int,
        out_width: int,
        kernel_size: int = 3,
        residual: bool = True,
        down_rate: Optional[int] = None,
        version: Optional[str] = None,
    ):
        super().__init__()
        self.d = down_rate
        self.residual = residual
        padding = 0 if kernel_size == 1 else 1

        if version == "light":  # uses less VRAM
            activation = nn.ReLU()
            self.conv = nn.Sequential(
                activation,
                nn.Conv2d(in_width, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, kernel_size, 1, padding),
            )
        else:  # for morphomnist
            activation = nn.GELU()
            self.conv = nn.Sequential(
                activation,
                nn.Conv2d(in_width, bottleneck, 1, 1),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
                activation,
                nn.Conv2d(bottleneck, out_width, 1, 1),
            )

        if self.residual and (self.d or in_width > out_width):
            self.width_proj = nn.Conv2d(in_width, out_width, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.residual:
            if x.shape[1] != out.shape[1]:
                x = self.width_proj(x)
            out = x + out
        if self.d:
            if isinstance(self.d, float):
                out = F.adaptive_avg_pool2d(out, int(out.shape[-1] / self.d))
            else:
                out = F.avg_pool2d(out, kernel_size=self.d, stride=self.d)
        return out


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # parse architecture
        stages = []
        for i, stage in enumerate(params["enc_arch"].split(",")):
            start = stage.index("b") + 1
            end = stage.index("d") if "d" in stage else None
            n_blocks = int(stage[start:end])

            if i == 0:  # define network stem
                if n_blocks == 0 and "d" not in stage:
                    print("Using stride=2 conv encoder stem.")
                    stem_width, stem_stride = params["widths"][1], 2
                    continue
                else:
                    stem_width, stem_stride = params["widths"][0], 1
                self.stem = nn.Conv2d(
                    params["input_channels"],
                    stem_width,
                    kernel_size=7,
                    stride=stem_stride,
                    padding=3,
                )
            stages += [(params["widths"][i], None) for _ in range(n_blocks)]
            if "d" in stage:  # downsampling block
                stages += [(params["widths"][i + 1], int(stage[stage.index("d") + 1]))]
        blocks = []
        for i, (width, d) in enumerate(stages):
            prev_width = stages[max(0, i - 1)][0]
            bottleneck = int(prev_width / params["bottleneck"])
            blocks.append(
                Block(prev_width, bottleneck, width, down_rate=d, version=params["vr"])
            )
        for b in blocks:
            b.conv[-1].weight.data *= np.sqrt(1 / len(blocks))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        x = self.stem(x)
        acts = {}
        for block in self.blocks:
            x = block(x)
            res = x.shape[2]
            if res % 2 and res > 1:  # pad if odd resolution
                x = F.pad(x, [0, 1, 0, 1])
            acts[x.size(-1)] = x
        return acts


class DecoderBlock(nn.Module):
    def __init__(self, params, in_width: int, out_width: int, resolution: int):
        super().__init__()
        bottleneck = int(in_width / params["bottleneck"])
        self.res = resolution
        self.stochastic = self.res <= params["z_max_res"]
        self.z_dim = params["z_dim"]
        self.cond_prior = json.loads(params["cond_prior"].lower())
        self.q_correction = json.loads(params["q_correction"].lower()) #convert to boolean value
        k = 3 if self.res > 2 else 1

        self.prior = Block(
            (in_width + params["context_dim"] if self.cond_prior else in_width),
            bottleneck,
            2 * self.z_dim + in_width,
            kernel_size=k,
            residual=False,
            version=params["vr"],
        )
        if self.stochastic:
            self.posterior = Block(
                2 * in_width + params["context_dim"],
                bottleneck,
                2 * self.z_dim,
                kernel_size=k,
                residual=False,
                version=params["vr"],
            )
        self.z_proj = nn.Conv2d(self.z_dim + params["context_dim"], in_width, 1)
        if not self.q_correction:  # for no posterior correction
            self.z_feat_proj = nn.Conv2d(self.z_dim + in_width, out_width, 1)
        self.conv = Block(
            in_width, bottleneck, out_width, kernel_size=k, version=params["vr"]
        )

    def forward_prior(
        self, z: Tensor, pa: Optional[Tensor] = None, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.cond_prior:
            z = torch.cat([z, pa], dim=1)
        z = self.prior(z)
        p_loc = z[:, : self.z_dim, ...]

        p_logscale = z[:, self.z_dim : 2 * self.z_dim, ...]
        p_features = z[:, 2 * self.z_dim :, ...]
        if t is not None:
            p_logscale = p_logscale + torch.tensor(t).to(z.device).log()
        return p_loc, p_logscale, p_features

    def forward_posterior(
        self, z: Tensor, x: Tensor, pa: Tensor, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        h = torch.cat([z, pa, x], dim=1)
        q_loc, q_logscale = self.posterior(h).chunk(2, dim=1)
        if t is not None:
            q_logscale = q_logscale + torch.tensor(t).to(z.device).log()
        return q_loc, q_logscale


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # parse architecture
        stages = []
        for i, stage in enumerate(params["dec_arch"].split(",")):
            res = int(stage.split("b")[0])
            n_blocks = int(stage[stage.index("b") + 1 :])
            stages += [(res, params["widths"][::-1][i]) for _ in range(n_blocks)]
        self.blocks = []
        for i, (res, width) in enumerate(stages):
            next_width = stages[min(len(stages) - 1, i + 1)][1]
            self.blocks.append(DecoderBlock(params, width, next_width, res))
        self._scale_weights()
        self.blocks = nn.ModuleList(self.blocks)
        # bias params
        self.all_res = list(np.unique([stages[i][0] for i in range(len(stages))]))
        bias = []
        for i, res in enumerate(self.all_res):
            if res <= params["bias_max_res"]:
                bias.append(
                    nn.Parameter(torch.zeros(1, params["widths"][::-1][i], res, res))
                )
        self.bias = nn.ParameterList(bias)
        self.cond_prior = json.loads(params["cond_prior"].lower())
        self.is_drop_cond = json.loads(params["is_drop_cond"].lower())

    def forward(
        self,
        parents: Tensor,
        x: Optional[Dict[int, Tensor]] = None,
        t: Optional[float] = None,
        abduct: bool = False,
        latents: List[Tensor] = [],
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        # learnt params for each resolution r
        bias = {r.shape[2]: r for r in self.bias}
        h = z = bias[1].repeat(parents.shape[0], 1, 1, 1)  # initial state
        # conditioning dropout: stochastic path (p_sto), deterministic path (p_det)
        p_sto, p_det = (
            self.drop_cond() if (self.training and self.cond_prior) else (1, 1)
        )

        stats = []
        for i, block in enumerate(self.blocks):
            res = block.res  # current block resolution, e.g. 64x64
            pa = parents[..., :res, :res].clone()  # select parents @ res

            # for morphomnist w/ conditioning dropout of y only, clean up later
            if self.is_drop_cond:
                pa_sto, pa_det = pa.clone(), pa.clone()
                pa_sto[:, 2:, ...] = pa_sto[:, 2:, ...] * p_sto
                pa_det[:, 2:, ...] = pa_det[:, 2:, ...] * p_det
            else:  # disabled otherwise
                pa_sto = pa_det = pa

            if h.size(-1) < res:  # upsample previous layer output
                b = bias[res] if res in bias.keys() else 0  # broadcasting
                h = b + F.interpolate(h, scale_factor=res / h.shape[-1])

            if block.q_correction:
                p_input = h  # current prior depends on previous posterior
            else:  # current prior depends on previous prior only, upsample previous prior latent z
                p_input = (
                    b + F.interpolate(z, scale_factor=res / z.shape[-1])
                    if z.size(-1) < res
                    else z
                )
            p_loc, p_logscale, p_feat = block.forward_prior(p_input, pa_sto, t=t) #compute prior p(zi|z<i)

            if block.stochastic:
                if x is not None:  # z_i ~ q(z_i | z_<i, x, pa_x)
                    q_loc, q_logscale = block.forward_posterior(h, x[res], pa, t=t) #compute q(zi |  z_<i, x, pa_x)
                    z = sample_gaussian(q_loc, q_logscale)
                    stat = dict(kl=gaussian_kl(q_loc, q_logscale, p_loc, p_logscale)) #kl between prior & posterior
                    if abduct:  # abduct exogenous noise
                        if block.cond_prior:  # z* if conditional prior
                            stat.update(
                                dict(
                                    z={"z": z, "q_loc": q_loc, "q_logscale": q_logscale}
                                )
                            )
                        else:  # z if exogenous prior
                            stat.update(dict(z=z))  # .detach() z if not cf training
                    stats.append(stat)
                else:
                    try:  # forward abducted latents
                        z = latents[i]
                        z = sample_gaussian(p_loc, p_logscale) if z is None else z
                    except:  # sample prior
                        z = sample_gaussian(p_loc, p_logscale)
                        if abduct and block.cond_prior:  # for abducting z*
                            stats.append(
                                dict(z={"p_loc": p_loc, "p_logscale": p_logscale})
                            )
            else:  # deterministic block
                z = p_loc
            h = h + p_feat  # merge prior features
            # h_i = h_<i + f(z_i, pa_x)
            h = h + block.z_proj(torch.cat([z, pa], dim=1))
            h = block.conv(h)

            if not block.q_correction:
                if (i + 1) < len(self.blocks):
                    # z independent of pa_x for next layer prior
                    z = block.z_feat_proj(torch.cat([z, p_feat], dim=1))
        return h, stats

    def _scale_weights(self):
        scale = np.sqrt(1 / len(self.blocks))
        for b in self.blocks:
            b.z_proj.weight.data *= scale
            b.conv.conv[-1].weight.data *= scale
            b.prior.conv[-1].weight.data *= 0.0

    @torch.no_grad()
    def drop_cond(self) -> Tuple[int, int]:
        opt = dist.Categorical(1 / 3 * torch.ones(3)).sample()
        if opt == 0:  # drop stochastic path
            p_sto, p_det = 0, 1
        elif opt == 1:  # drop deterministic path
            p_sto, p_det = 1, 0
        elif opt == 2:  # keep both
            p_sto, p_det = 1, 1
        return p_sto, p_det


class DGaussNet(nn.Module):
    def __init__(self, params):
        super(DGaussNet, self).__init__()
        self.x_loc = nn.Conv2d(
            params["widths"][0], params["input_channels"], kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv2d(
            params["widths"][0], params["input_channels"], kernel_size=1, stride=1
        )

        if params["input_channels"] == 3:
            self.channel_coeffs = nn.Conv2d(params["widths"][0], 3, kernel_size=1, stride=1)

        if params["std_init"] > 0:  # if std_init=0, random init weights for diag cov
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, np.log(params["std_init"]))

            covariance = params["x_like.split"]("_")[0]
            if covariance == "fixed":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = False
            elif covariance == "shared":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = True
            elif covariance == "diag":
                self.x_logscale.weight.requires_grad = True
                self.x_logscale.bias.requires_grad = True
            else:
                NotImplementedError(params["x_like"] + " is not implemented.")

    def forward(
        self, h: Tensor, x: Optional[Tensor] = None, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=EPS)

        # for RGB inputs
        if hasattr(self, "channel_coeffs"):
            coeff = torch.tanh(self.channel_coeffs(h))
            if x is None:  # inference
                # loc = loc + logscale.exp() * torch.randn_like(loc)  # random sampling
                f = lambda x: torch.clamp(x, min=-1, max=1)
                loc_red = f(loc[:, 0, ...])
                loc_green = f(loc[:, 1, ...] + coeff[:, 0, ...] * loc_red)
                loc_blue = f(
                    loc[:, 2, ...]
                    + coeff[:, 1, ...] * loc_red
                    + coeff[:, 2, ...] * loc_green
                )
            else:  # training
                loc_red = loc[:, 0, ...]
                loc_green = loc[:, 1, ...] + coeff[:, 0, ...] * x[:, 0, ...]
                loc_blue = (
                    loc[:, 2, ...]
                    + coeff[:, 1, ...] * x[:, 0, ...]
                    + coeff[:, 2, ...] * x[:, 1, ...]
                )

            loc = torch.cat(
                [loc_red.unsqueeze(1), loc_green.unsqueeze(1), loc_blue.unsqueeze(1)],
                dim=1,
            )

        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        return loc, logscale

    def approx_cdf(self, x: Tensor) -> Tensor:
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h: Tensor, x: Tensor) -> Tensor:
        loc, logscale = self.forward(h, x)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(
        self, h: Tensor, return_loc: bool = False, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()



class MmnistCondHVAE(CondHVAE):

    def __init__(self, attr_size, params, name="image_hvae"):

        params["context_dim"] = sum(attr_size.values())
        self.cf_fine_tune = json.loads(params["cf_fine_tune"].lower())  
        self.evaluate = json.loads(params["evaluate"].lower())
        self.load_ckpt = json.loads(params["load_pretrained_ckpt"].lower())
        self.name = name
        encoder = Encoder(params)
        decoder = Decoder(params)
        likelihood = DGaussNet(params)
      

        super().__init__(encoder, decoder, likelihood, params,  self.load_ckpt,
                         self.cf_fine_tune, self.evaluate, self.name)
        
        if not self.cf_fine_tune:
            self.apply(init_bias)



if __name__ == "__main__":

    attribute_size = {
        "thickness": 1,
        "intensity": 1,
        "digit": 10
    }


    config_file = "../../methods/deepscm/configs/morphomnist_hvae_config.json"
    with open(config_file, 'r') as f:
        config = load(f)

    params = config["mechanism_models"]["image"]["params"]

    train_set = MorphoMNISTLike(attribute_size=attribute_size, train=True)

    tr_data_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=7)
    iterator = iter(tr_data_loader)
    batch = next(iterator)
    x , attrs = batch


    attrs =attrs[..., None, None].repeat(1, 1, *(32,) * 2)

    #x = torch.randn(1, 1, 32, 32)
    #x = torch.clamp(x , -1, 1)
    #attrs = torch.randn(1, 12)[..., None, None].repeat(1, 1, *(32,) * 2)


    model = MmnistCondHVAE(attribute_size, params, name="hvae")
    print(model.name)

   # conv = model.decoder.blocks[0].prior.conv
   # inp = torch.zeros(1, 256, 1, 1)
    #out1 = conv(inp)

    out = model(x, attrs)
    print(out)
    #print(conv)
    #out = model(x, attrs)
    #print(out)

