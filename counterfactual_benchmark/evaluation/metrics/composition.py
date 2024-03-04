import numpy as np
import sys
sys.path.append("../../")
from evaluation.embeddings.vgg import vgg_normalize
from models.utils import rgbify

def composition(factual_batch, unnormalize_fn, method, cycles=[1, 10], device='cuda', embedding=None, embedding_model=None):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    cond = factual_batch["intensity"] if "intensity" in factual_batch else None
    images = [factual_batch["image"]]

    for _ in range(max(cycles)):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(counterfactual_batch["image"])
        factual_batch = counterfactual_batch

    composition_scores = l1_distance(images, cond, steps=cycles, embedding=embedding, embedding_model=embedding_model, unnormalize_fn=unnormalize_fn)

    # stack images for all cycles
    all_images = np.concatenate([unnormalize_fn(image, "image").cpu().numpy() for image in images], axis=3)
    return composition_scores, all_images

def l1_distance(images, cond, steps, embedding, embedding_model, unnormalize_fn):
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

    distances = {}
    for step in steps:
        if embedding == "lpips":
            distances[step] = np.array([embedding_fn(images[step], images[0])])
        else:
            distances[step] = np.mean(np.abs(embedding_fn(images[step]) - embedding_fn(images[0])), axis=(1,2,3) if embedding is None else 1)

    return distances
