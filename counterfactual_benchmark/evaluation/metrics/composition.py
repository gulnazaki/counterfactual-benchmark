import matplotlib.pyplot as plt
import numpy as np
from datasets.morphomnist.dataset import unnormalize

def composition(factual_batch, num_batch, method, cycles=10):
    images = [unnormalize(factual_batch["image"], name="image")]
    for _ in range(cycles):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(unnormalize(counterfactual_batch["image"], name="image"))
        factual_batch = counterfactual_batch
    # plot all images
    all_images = np.concatenate(images, axis=3)
    plt.imsave("composition_samples/composition_sample{}.png".format(num_batch), all_images[0][0], cmap='gray')

    # TODO loop on different embeddings
    return l1_distance(images, steps=cycles) # add more distances

def l1_distance(images, steps):
    final_img = (images[steps]).numpy()
    init_img = images[0].numpy()

    return np.mean(np.abs(final_img - init_img))
