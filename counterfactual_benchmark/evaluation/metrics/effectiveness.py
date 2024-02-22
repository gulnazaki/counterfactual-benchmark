from torchmetrics.classification import BinaryAccuracy

def effectiveness(counterfactual_batch, unnormalize_fn, predictors):
   # print(counterfactual_batch["intensity"].shape)
    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents
    predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                   if key=="thickness"
                   else clfs(counterfactual_batch["image"]) for key , clfs in predictors.items()} #predicted values

    if "digit" in list(targets.keys()):
        result = {key:(unnormalize_fn(targets[key], key) - unnormalize_fn(predictions[key], key)).abs().mean().detach().cpu().numpy()
              if key!="digit" else  (targets[key].argmax(-1).cpu().numpy() == predictions[key].argmax(-1).cpu().numpy()).mean()
              for key in targets}

    else: #celeba attributes
        result = {key: BinaryAccuracy()(predictions[key].detach().cpu(), targets[key].detach().cpu()) for key in targets}

    return result
