import numpy as np
from ..embeddings.vgg import vgg

def composition(factual_batch, unnormalize_fn, method, cycles=[1, 10], device='cuda', embedding=None, pretrained=False):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    images = [unnormalize_fn(factual_batch["image"], name="image")]

    for _ in range(max(cycles)):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(unnormalize_fn(counterfactual_batch["image"], name="image"))
        factual_batch = counterfactual_batch

    # TODO loop on different embeddings
    composition_scores = l1_distance(images, steps=cycles, embedding=embedding, pretrained=pretrained) # add more distances

    # stack images for all cycles
    all_images = np.concatenate([image.cpu().numpy() for image in images], axis=3)
    return composition_scores, all_images

def l1_distance(images, steps, embedding, pretrained):
    if embedding is None:
        embedding_fn = lambda x: x.cpu().numpy()
    elif embedding == "vgg":
        model = vgg(pretrained)
        embedding_fn = lambda x: model(x).cpu().numpy()
    else:
        print(f"WARNING: Invalid embedding: {embedding}, only 'vgg' is supported. "
              "Computing l1-distance on image space...")
        embedding = None
        embedding_fn = lambda x: x.cpu().numpy()

    distances = {}
    for step in steps:
        distances[step] = np.mean(np.abs(embedding_fn(images[step]) - embedding_fn(images[0])), axis=(1,2,3) if embedding is None else 1)

    return distances
