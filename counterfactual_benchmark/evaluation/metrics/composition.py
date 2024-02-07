import numpy as np
from datasets.morphomnist.dataset import unnormalize

def composition(factual_batch, num_batch, method, cycles=10):
    images = [unnormalize(factual_batch["image"], name="image")]
    for _ in range(cycles):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(unnormalize(counterfactual_batch["image"], name="image"))
        factual_batch = counterfactual_batch

    # TODO loop on different embeddings
    composition_scores = l1_distance(images, steps=cycles) # add more distances

    # stack images for all cycles
    all_images = np.concatenate(images, axis=3)
    return composition_scores, all_images

def l1_distance(images, steps):
    final_img = (images[steps]).numpy()
    init_img = images[0].numpy()

    return np.mean(np.abs(final_img - init_img), axis=(1,2,3))
