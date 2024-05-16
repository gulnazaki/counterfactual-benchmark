from torchmetrics.classification import BinaryF1Score
import torch.nn as nn
import torch

def effectiveness(counterfactual_batch, unnormalize_fn, predictors, dataset):

    #select the counterfactual  parents
    targets = {key: value for key, value in counterfactual_batch.items() if key != "image"}
    if dataset == 'adni':
        predictions = {}
        for key, clfs in predictors.items():
            if clfs.cond_atts is not None:
                cond = torch.cat([counterfactual_batch[att] for att in clfs.cond_atts], dim=1)
                predictions[key] = clfs(counterfactual_batch["image"], cond) if clfs.image_as_input else clfs(cond)
            else:
                predictions[key] = clfs(counterfactual_batch["image"])

        result = {key: clfs.metric(torch.sigmoid(predictions[key]), targets[key]).cpu().detach().numpy() for key, clfs in predictors.items()}
    else:
        predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                    if key=="thickness"
                    else nn.Sigmoid()(clfs(counterfactual_batch["image"])) for key , clfs in predictors.items()} #predicted values

        if "digit" in list(targets.keys()):
            result = {key:(unnormalize_fn(targets[key], key) - unnormalize_fn(predictions[key], key)).abs().mean().detach().cpu().numpy()
                if key!="digit" else  (targets[key].argmax(-1) == predictions[key].argmax(-1)).sum().detach().cpu().numpy() / predictions[key].shape[0]
                for key in targets}

        else: #celeba attributes
            result = {key: BinaryF1Score(threshold=0.5).to('cuda')(predictions[key], targets[key]).cpu() for key in targets}

    return result
