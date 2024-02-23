import numpy as np

def composition(factual_batch, unnormalize_fn, method, cycles=[1, 10], device='cuda'):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}
    images = [unnormalize_fn(factual_batch["image"].cpu(), name="image")]

    for _ in range(max(cycles)):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(unnormalize_fn(counterfactual_batch["image"].cpu(), name="image"))
        factual_batch = counterfactual_batch

    # TODO loop on different embeddings
    composition_scores = l1_distance(images, steps=cycles) # add more distances

    # stack images for all cycles
    all_images = np.concatenate(images, axis=3)
    return composition_scores, all_images

def l1_distance(images, steps):
    distances = {}
    for step in steps:
        distances[step] = np.mean(np.abs(images[step].numpy() - images[0].numpy()), axis=(1,2,3))

    return distances
