import matplotlib.pyplot as plt
import numpy as np
import torch

def composition(factual_batch, num_batch, method, cycles=10):
    images = [unnormalize_image(factual_batch["image"])]
    for _ in range(cycles):
        abducted_noise = method.encode(**factual_batch)
        counterfactual_batch = method.decode(**abducted_noise)
        images.append(unnormalize_image(counterfactual_batch["image"]))
        factual_batch = counterfactual_batch
    # plot all images
    all_images = np.concatenate(images, axis=3)
    plt.imsave("composition_samples/composition_sample{}.png".format(num_batch), all_images[0][0], cmap='gray')

    # loop on different embeddings
    return l1_distance(images, steps=cycles) # add more distances

def l1_distance(images, steps):
    final_img = (images[steps]).numpy()
    init_img = images[0].numpy()

    return np.mean(np.abs(final_img - init_img))

def unnormalize_image(img):
    return (img + 1.0) * 127.5
    # norm_params = torch.load("../../datasets/morphomnist/data/norm_params.pt")
    # img_m = norm_params["img_mean"]
    # img_s = norm_params["img_std"]

    # return (img * img_s) + img_m

