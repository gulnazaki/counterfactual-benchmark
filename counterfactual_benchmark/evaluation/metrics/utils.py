import matplotlib.pyplot as plt
import numpy as np
import os


def save_image(img, path):
    if img.shape[0] == 3:
        plt.imsave(path, img.transpose(1, 2, 0))
    else:
        plt.imsave(path, img[0], cmap='gray')
    return

def save_selected_images(images, scores, save_dir, lower_better=True, n_best=10, n_worst=10, n_median=10):
    sort_ids = np.argsort(scores)
    images_sorted = images[sort_ids] if lower_better else images[sort_ids][::-1]

    for i in range(n_best):
        save_image(images_sorted[i], os.path.join(save_dir, f"best_{i}.png"))

    for i in range(n_worst):
        save_image(images_sorted[-i], os.path.join(save_dir, f"worst_{i}.png"))

    total = scores.shape[0]
    for i in range(n_median):
        save_image(images_sorted[(total//2) - (n_median//2) + i], os.path.join(save_dir, f"median_{i}.png"))

    return

def save_plots(images, fig_idx):
    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
    titles = ["factual", "do(thickness)", "do(intensity)", "do(digit)"]
    for i, img in enumerate(images):
        if img.shape[0] == 3:
            axs[i].imshow(img.transpose(1, 2, 0))
        else:
            axs[i].imshow(img[0], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')  # Turn off axis
        plt.tight_layout()
     #   plt.title()

    plt.savefig("qualitative_samples/images_plot_{}.png".format(fig_idx))

    plt.close()
    return