import torch
import numpy as np
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike
from evaluation.metrics.composition import composition
from evaluation.metrics.coverage_density import coverage_density
from evaluation.embeddings.vgg import vgg

from datasets.transforms import ReturnLabelsTransform


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike
}



def evaluate_coverage_density(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False, num_workers=7)
  
    counterfactuals = []
    factuals = []  
    for batch in test_data_loader:
        counterfactual_batch =  produce_counterfactuals(batch, scm, do_parent='thickness', intervention_source=real_set)
        counterfactuals.append(counterfactual_batch['image'])
        batch = (counterfactual_batch['image']).numpy()
    for batch in real_data_loader:
        factuals.append(batch['image'])
        
    return coverage_density(factuals, counterfactuals, k = 5, embedding_fn=vgg, pretrained=True)




def evaluate_composition(test_set: Dataset, batch_size: int, cycles: int, scm: nn.Module):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    # composition
    composition_scores = []
    for i, factual_batch in enumerate(tqdm(test_data_loader)):
        composition_scores.append(composition(factual_batch, i, method=scm, cycles=cycles))
    composition_score = np.mean(composition_scores)
    print("Composition score:", composition_score)

    return composition_score

    # # get "true" counterfactuals
    # counterfactuals = []
    # for factual_batch in test_data_loader:
    #     for do_pa in attributes:
    #         idx = torch.randperm(train_set[do_pa].shape[0])
    #         interventions = train_set[do_pa].clone()[idx][:self.batch_size]
    #         abducted_noise = method.encode(factual_batch, factual_batch["attrs"])
    #         counterfactual_batch, counterfactual_parents = method.decode(abducted_noise, interventions)
    #         counterfactuals.append((counterfactual_batch, counterfactual_parents))
    # # effectiveness
    # effectiveness(counterfactuals, predictors)

    # # coverage & density
    # coverage_density(train_set["image"], counterfactuals[0])



def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, intervention_source: Dataset):

    batch_size, _ , _ , _ = factual_batch["image"].shape
    idxs = torch.randperm(len(intervention_source))[:batch_size] # select random indices from train set to perform interventions
   
   
    #update with the counterfactual parent
    interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1)}
    
    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch
    


def evaluate_effectiveness():
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    config_file = "configs/morphomnist_config.json"
    with open(config_file, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attributes = config["causal_graph"]["image"]

    models = {}
    for variable in config["causal_graph"].keys():
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(name=variable, params=model_config["params"], attrs=attributes)

        models[variable] = model

    scm = SCM(ckpt_path=config["checkpoint_dir"],
              graph_structure=config["causal_graph"],
              **models)
    
    dataset = config["dataset"]
    data_class = dataclass_mapping[dataset]
    transform = ReturnLabelsTransform(attributes=attributes, image_name="image")
    
    train_set = data_class(attributes=attributes, train=True, columns=attributes, transform=transform)
    test_set = data_class(attributes=attributes, train=False, columns=attributes, transform=transform)

    evaluate_coverage_density(real_set=train_set, test_set=test_set, batch_size=1, scm=scm) 
    evaluate_composition(test_set, batch_size=256, cycles=10, scm=scm)


    #########################################################################################################################
    ## just test code for the produced counterfactuals -> may delete later 
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=7)
    iterator = iter(test_data_loader)
    batch = next(iterator)
    counterfactuals = produce_counterfactuals(batch, scm, do_parent="thickness", intervention_source=train_set)
    
    cf_image = counterfactuals["image"].squeeze(0).squeeze(0).numpy()

    plt.imsave("cf_img.png", cf_image, cmap='gray')



