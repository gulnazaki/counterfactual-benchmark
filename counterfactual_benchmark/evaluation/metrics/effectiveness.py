from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
import torch.nn as nn

def effectiveness(counterfactual_batch, unnormalize_fn, predictors):

    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents
    predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                   if key=="thickness"
                   else nn.Sigmoid()(clfs(counterfactual_batch["image"])) for key , clfs in predictors.items()} #predicted values

    if "digit" in list(targets.keys()):
        result = {key:(unnormalize_fn(targets[key], key) - unnormalize_fn(predictions[key], key)).abs().mean().cpu().numpy()
              if key!="digit" else  (targets[key].argmax(-1) == predictions[key].argmax(-1)).sum().cpu().numpy() / predictions[key].shape[0]
              for key in targets}

    else: #celeba attributes
        result = {key: BinaryF1Score(threshold=0.5).to('cuda')(predictions[key], targets[key]).cpu() for key in targets}

    return result
