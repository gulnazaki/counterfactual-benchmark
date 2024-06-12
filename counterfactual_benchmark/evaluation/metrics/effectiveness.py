from torchmetrics.classification import BinaryF1Score
import torch.nn as nn
import torch
def effectiveness(counterfactual_batch, unnormalize_fn, predictors, dataset):
   # print(counterfactual_batch["intensity"].shape)
    targets = {key: value for key , value in counterfactual_batch.items() if key!="image"} #select the counterfactual  parents

   # print(targets.keys())
    if dataset == "celeba":
        predictions = {}
        for key, clfs in predictors.items():
            if clfs.conditions[key] is not None:
                cond = torch.cat([counterfactual_batch[att] for att in clfs.conditions[key]], dim=1)
                predictions[key] = clfs(counterfactual_batch["image"], cond)

            else:
                predictions[key] = clfs(counterfactual_batch["image"])
           # print(predictions)

       # print(predictions['Young'])

        result = {key: BinaryF1Score(threshold=0.5).to("cuda")(predictions[key], targets[key]).cpu().detach().numpy() for key in predictions}

    elif dataset == "morphomnist":

        predictions = {key: clfs(counterfactual_batch["image"], counterfactual_batch["intensity"])
                   if key=="thickness"
                   else nn.Sigmoid()(clfs(counterfactual_batch["image"])) for key , clfs in predictors.items()} #predicted values

        if "digit" in list(targets.keys()):
            result = {key:(unnormalize_fn(targets[key], key) - unnormalize_fn(predictions[key], key)).abs().mean().cpu().numpy()
              if key!="digit" else  (targets[key].argmax(-1) == predictions[key].argmax(-1)).sum().cpu().numpy() / predictions[key].shape[0]
              for key in targets}

    elif dataset == "adni":
        predictions = {}
        for key, clfs in predictors.items():
            if clfs.cond_atts is not None:
                cond = torch.cat([counterfactual_batch[att] for att in clfs.cond_atts], dim=1)
                predictions[key] = clfs(counterfactual_batch["image"], cond) if clfs.image_as_input else clfs(cond)
            else:
                predictions[key] = clfs(counterfactual_batch["image"])

        result = {key: clfs.metric(torch.sigmoid(predictions[key]), targets[key]).cpu().detach().numpy() for key, clfs in predictors.items()}

    return result
