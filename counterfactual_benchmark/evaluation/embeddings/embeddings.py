from .vgg import vgg, vgg_normalize
from .classifier_embeddings import ClassifierEmbeddings
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import numpy as np
import sys
import torch
from functools import partial
sys.path.append("../../")
from models.utils import rgbify
from models.vaes import MmnistCondVAE, CelebaCondVAE
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip

def get_embedding_model(embedding, pretrained_vgg, classifier_config=None):
    if embedding == "vgg":
        return vgg(pretrained_vgg)
    elif embedding == "clfs":
        return ClassifierEmbeddings(classifier_config)
    elif embedding == "vae":
        if 'morphomnist' in classifier_config:
            params = {'latent_dim': 16, 'hidden_dim': 128, 'n_chan': [1, 32, 32, 32], 'beta': 1, 'lr': 1e-3, 'weight_decay': 0.01, 'fixed_logvar': "False"}
            attribute_size = {
                "thickness": 1,
                "intensity": 1,
                "digit": 10
            }
            model = MmnistCondVAE(params, attribute_size, unconditional=True).eval().to('cuda')
            model.load_state_dict(torch.load('../../methods/deepscm/checkpoints/trained_scm/uncond_image_vae-epoch=269.ckpt', map_location=torch.device('cuda'))["state_dict"])
        else:
            params = {'latent_dim': 16, 'hidden_dim': 256, 'n_chan': [3, 32, 64, 128, 256, 256], 'beta': 5, 'lr': 0.0005, 'weight_decay': 0, 'fixed_logvar': "False"}
            attribute_size = {
                "Smiling": 1,
                "Eyeglasses": 1
            }
            model = CelebaCondVAE(params, attribute_size, unconditional=True).eval().to('cuda')
            model.load_state_dict(torch.load('../../methods/deepscm/checkpoints_celeba/trained_scm/uncond_image_vae-epoch=44.ckpt', map_location=torch.device('cuda'))["state_dict"])
            return model
        return model
    elif embedding == "lpips":
        return LPIPS(net_type='vgg', normalize=True).to('cuda')
    elif embedding == "clip":
        model = clip.load("ViT-B/32", device='cuda')[0]
        def _transform(n_px):
            return Compose([
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return model, _transform(model.visual.input_resolution)
    else:
        return None

def get_embedding_fn(embedding, unnormalize_fn, embedding_model):
    if embedding is None:
        return partial(unnormalize_embedding_fn, unnormalize_fn)
    elif embedding == "vgg":
        return partial(vgg_embedding_fn, embedding_model)
    elif embedding == "clfs":
        return partial(clfs_embedding_fn, embedding_model)
    elif embedding == "vae":
        return partial(vae_embedding_fn, embedding_model)
    elif embedding == "lpips":
        return partial(lpips_embedding_fn, embedding_model)
    elif embedding == "clip":
        return partial(clip_embedding_fn,  embedding_model)
    else:
        exit(f"Invalid embedding: {embedding}")

def unnormalize_embedding_fn(unnormalize_fn, x, _):
    return unnormalize_fn(x, "image").cpu().numpy()

def vgg_embedding_fn(embedding_model, x, _):
    return embedding_model(vgg_normalize(rgbify(x, normalized=True), to_0_1=False)).detach().cpu().numpy()

def clfs_embedding_fn(embedding_model, x, cond, skip_attribute=None):
    return embedding_model(x, cond, only_intensity=True, skip_attribute=skip_attribute).detach().cpu().numpy()

def vae_embedding_fn(embedding_model, x, cond):
    mu, logvar = embedding_model.encoder(x, cond)
    return np.transpose([mu.detach().cpu(), logvar.detach().cpu()], (1, 0, 2))

def lpips_embedding_fn(embedding_model, x, y):
    return embedding_model(rgbify(x, normalized=True), rgbify(y, normalized=True)).detach().cpu().numpy()

def clip_embedding_fn(embedding_model, x, _):
    model, preprocess = embedding_model
    image = preprocess(rgbify(x, normalized=False))
    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features.detach().cpu().numpy()
