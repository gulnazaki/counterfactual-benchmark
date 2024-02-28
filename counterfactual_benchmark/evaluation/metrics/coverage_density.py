from ..embeddings.vgg import vgg, vgg_normalize
from .prdc import compute_prdc
from ...models.utils import rgbify
import torchvision.transforms as T
import torch
import numpy as np
from tqdm import tqdm
import os


def coverage_density(real_images, generated_images, k = 5, embedding_fn=vgg, pretrained=True, feat_path=None):
    transform224 = T.Resize(size = (224,224), antialias=True)

    model = embedding_fn(pretrained)

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
            rgb_batch = rgbify(image)
            input = vgg_normalize(transform224(rgb_batch), to_0_1=False)
            if torch.cuda.is_available():
                input = input.to("cuda")
            feat = model(input).cpu().detach().numpy()
            features[type_].append(feat)
        features[type_] = np.concatenate(features[type_])
        if type_ == "real" and feat_path is not None:
            print(f"Saving real features for coverage_density to {feat_path}")
            np.save(feat_path, features[type_])

    metrics = compute_prdc(features["real"], features["generated"], k)

    print (f"Coverage: {round(metrics['coverage'], 3)}")
    print (f"Density: {round(metrics['density'], 3)}")
    print (f"Precision: {round(metrics['precision'], 3)}")
    print (f"Recall: {round(metrics['recall'], 3)}")

    return features["real"], features["generated"]
