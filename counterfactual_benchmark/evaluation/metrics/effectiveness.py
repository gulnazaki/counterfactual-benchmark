import matplotlib.pyplot as plt
import numpy as np
import torch


def effectiveness(counterfactual_batch, predictors):
   # print(counterfactual_batch["intensity"].shape)
    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents
    predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"]) 
                   if key=="thickness"
                   else clfs(counterfactual_batch["image"]) for key , clfs in predictors.items()} #predicted values

    result = {key:(targets[key] - predictions[key]).abs().mean().detach().numpy() for key in targets}

    return result




'''
def l1_distance(all_images, steps):
    #compute l1_distance on unnormalize images
    final_img = unnormalize_image(all_images[steps]).numpy()
    init_img = unnormalize_image(all_images[0]).numpy()

    return np.mean(np.abs(final_img - init_img))

def unnormalize_image(img):
    norm_params = torch.load("../../datasets/morphomnist/data/norm_params.pt")
    img_m = norm_params["img_mean"]
    img_s = norm_params["img_std"]

    return (img * img_s) + img_m 

    '''