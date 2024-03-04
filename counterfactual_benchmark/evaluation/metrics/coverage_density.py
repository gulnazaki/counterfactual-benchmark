import torchvision.transforms as T
import torch
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append("../../")
from evaluation.embeddings.vgg import vgg_normalize
from models.utils import rgbify

def vgg_features(real_images, generated_images, embedding, embedding_model=None, feat_path=None, unnormalize_fn=None):
    if embedding is None:
        embedding_fn = lambda x: unnormalize_fn(x, "image").cpu().numpy()
    elif embedding == "vgg":
        embedding_fn = lambda x: embedding_model(vgg_normalize(rgbify(x, normalized=True), to_0_1=False)).detach().cpu().numpy()
    elif embedding == "clfs":
        embedding_fn = lambda x: embedding_model(x, cond, only_intensity=True).detach().cpu().numpy()
    elif embedding == "lpips":
        embedding_fn = lambda x, y: embedding_model(rgbify(x, normalized=True), rgbify(y, normalized=True)).detach().cpu().numpy()
    else:
        exit(f"Invalid embedding: {embedding}")

    images = {"real": real_images,
              "generated": generated_images}
    features = {"real": [],
                "generated": []}

    for type_ in images:
        if type_ == "real" and feat_path is not None and os.path.isfile(feat_path):
            print(f"Loading real features for coverage_density from {feat_path}")
            features[type_] = np.load(feat_path)
            continue

        for image in tqdm(images[type_]):
            feat = embedding_fn(image).cpu().detach().numpy()
            features[type_].append(feat)
        features[type_] = np.concatenate(features[type_])
        if type_ == "real" and feat_path is not None:
            print(f"Saving real features for coverage_density to {feat_path}")
            np.save(feat_path, features[type_])

    return features["real"], features["generated"]
