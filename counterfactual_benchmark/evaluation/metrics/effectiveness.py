
def effectiveness(counterfactual_batch, unnormalize_fn, predictors):
   # print(counterfactual_batch["intensity"].shape)
    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents
    predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                   if key=="thickness"
                   else clfs(counterfactual_batch["image"]) for key , clfs in predictors.items()} #predicted values

#acc = (targets['digit'].argmax(-1).numpy() == preds['digit'].argmax(-1).numpy()).mean()

    result = {key:(unnormalize_fn(targets[key], key) - unnormalize_fn(predictions[key], key)).abs().mean().detach().numpy()
              if key!="digit" else  (targets[key].argmax(-1).numpy() == predictions[key].argmax(-1).numpy()).mean()
              for key in targets}

    return result
