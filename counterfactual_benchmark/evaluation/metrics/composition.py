import matplotlib.pyplot as plt
import numpy as np

def composition(factual_batch, method, cycles=10):
    images = [factual_batch["image"]]
    for _ in range(cycles):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(counterfactual_batch["image"])
        factual_batch = counterfactual_batch

    # plot all images
    all_images = np.concatenate(images, axis=3)
    plt.imsave("composition_sample.png", all_images[0][0], cmap='gray')

    # loop on different embeddings
    return l1_distance(images[0], images[-1]) # add more distances

def l1_distance(image1, image2):
    return np.sum(np.abs(image1 - image2))
