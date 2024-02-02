from datasets.morphomnist.dataset import unnormalize

def effectiveness(counterfactual_batch, predictors):
   # print(counterfactual_batch["intensity"].shape)
    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents
    predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                   if key=="thickness"
                   else clfs(counterfactual_batch["image"]) for key , clfs in predictors.items()} #predicted values

    result = {key:(targets[key] - predictions[key]).abs().mean().detach().numpy() for key in targets}

    return result
