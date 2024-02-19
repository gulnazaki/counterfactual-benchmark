from ..embeddings.vgg import vgg
from .prdc import compute_prdc
import torchvision.transforms as T
import torch
import numpy as np
from tqdm import tqdm


def coverage_density(real_images, generated_images, k = 5, embedding_fn=vgg, pretrained=True):
    transform224 = T.Resize(size = (224,224), antialias=True)

    model = embedding_fn(pretrained)

    images = {"real": real_images,
              "generated": generated_images}
    features = {"real": [],
                "generated": []}

    for type_ in images:
        for image in tqdm(images[type_]):
            rgb_batch = np.repeat(image, 3, axis=1) if image.shape[1] == 1 else image
            input = transform224(rgb_batch)
            if torch.cuda.is_available():
                input = input.to("cuda")
            feat = model(input).cpu().detach().numpy()
            features[type_].append(feat)
        features[type_] = np.concatenate(features[type_])

    metrics = compute_prdc(features["real"], features["generated"], k)

    print ('Coverage: ', metrics['coverage'])
    print ('Density: ', metrics['density'])
    print ('Precision: ', metrics['precision'])
    print ('Recall: ', metrics['recall'])