import matplotlib.pyplot as plt
import numpy as np
import os


def save_selected_images(images, scores, save_dir, lower_better=True, n_best=10, n_worst=10, n_median=10):
    sort_ids = np.argsort(scores)
    images_sorted = images[sort_ids] if lower_better else images[sort_ids][::-1]

    for i in range(n_best):
        plt.imsave(os.path.join(save_dir, f"best_{i}.png"), images_sorted[i][0], cmap='gray')

    for i in range(n_worst):
        plt.imsave(os.path.join(save_dir, f"worst_{i}.png"), images_sorted[-i][0], cmap='gray')

    total = scores.shape[0]
    for i in range(n_median):
        plt.imsave(os.path.join(save_dir, f"median_{i}.png"), images_sorted[(total//2) - (n_median//2) + i][0], cmap='gray')
