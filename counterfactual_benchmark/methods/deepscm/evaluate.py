import torch
import numpy as np
from typing import Dict, Tuple, List
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append("../../")

from models.classifiers.classifier import Classifier
from datasets.morphomnist.dataset import MorphoMNISTLike
from evaluation.metrics.composition import composition
from evaluation.metrics.effectiveness import effectiveness
from evaluation.metrics.utils import save_selected_images, save_plots

from evaluation.metrics.coverage_density import coverage_density
from evaluation.embeddings.vgg import vgg

from datasets.transforms import ReturnDictTransform


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike
}

def produce_qualitative_samples(dataset, scm, parents, intervention_source):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    for i , batch in tqdm(enumerate(data_loader)):
        if i % 500 == 0:
            res = [batch["image"].squeeze(0).squeeze(0)]
            #dataset[i]["image"]  = dataset[i]["image"].unsqueeze(0)
            #print(dataset[i]["image"].shape)

            for do_parent in parents:
                counterfactual = produce_counterfactuals(batch, scm, do_parent, intervention_source)
                res.append(counterfactual["image"].squeeze(0).squeeze(0))

            save_plots(res, i)
    return



def evaluate_coverage_density(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False, num_workers=7)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    counterfactual_images = []
    for factual_batch in tqdm(test_data_loader):
        counterfactual_batch =  produce_counterfactuals(factual_batch, scm, do_parent='thickness', intervention_source=real_set)
        counterfactual_images.append(counterfactual_batch['image'])

    real_images = [batch["image"] for batch in real_data_loader]
    return coverage_density(real_images, generated_images=counterfactual_images, k = 5, embedding_fn=vgg, pretrained=True)


def evaluate_composition(test_set: Dataset, batch_size: int, cycles: int, scm: nn.Module):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    composition_scores = []
    images = []
    for i, factual_batch in enumerate(tqdm(test_data_loader)):
        score_batch, image_batch = composition(factual_batch, i, method=scm, cycles=cycles)
        composition_scores.append(score_batch)
        images.append(image_batch)

    images = np.concatenate(images)
    composition_scores = np.concatenate(composition_scores)

    save_selected_images(images, composition_scores, save_dir="composition_samples", lower_better=True)

    composition_score = np.mean(composition_scores)
    print("Average composition score:", composition_score)

    return composition_score


def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, intervention_source: Dataset):

    batch_size, _ , _ , _ = factual_batch["image"].shape
    idxs = torch.randperm(len(intervention_source))[:batch_size] # select random indices from train set to perform interventions

   # print(idxs)
    #update with the counterfactual parent

    interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1)
                     if do_parent!="digit" else torch.cat([intervention_source[id][do_parent].unsqueeze(0) for id in idxs])}


    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch


def evaluate_effectiveness(test_set: Dataset, batch_size:int , scm: nn.Module, attributes: List, do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier]):

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    for factual_batch in tqdm(test_data_loader):
        counterfactuals = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source)
        e_score = effectiveness(counterfactuals, predictors)

        for attr in attributes:
            effectiveness_scores[attr].append(e_score[attr])

    effectiveness_score = {key  : np.mean(score) for key, score in effectiveness_scores.items()}

    print("Effectiveness score " + "do("+do_parent+"):", effectiveness_score)

    return effectiveness_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, coverage_density] or use 'all'.",
                        default=["all"])
    # parser.add_argument("--config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

   # torch.manual_seed(42)
    config_file = "configs/morphomnist_hvae_config.json"
    config_file_cls = "configs/morphomnist_classifier_config.json"

    with open(config_file_cls, 'r') as f1:
        config_cls = load(f1)

    with open(config_file, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(name=variable, params=model_config["params"], attr_size=attribute_size)

        models[variable] = model

    scm = SCM(checkpoint_dir=config["checkpoint_dir"],
              graph_structure=config["causal_graph"],
              **models)

    dataset = config["dataset"]
    data_class = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    train_set = data_class(attribute_size, train=True, transform=transform)
    test_set = data_class(attribute_size, train=False, transform=transform)

   #  produce_qualitative_samples(dataset=test_set, scm=scm, parents=list(attribute_size.keys()), intervention_source=train_set)


    if "composition" in args.metrics or "all" in args.metrics:
        evaluate_composition(test_set, batch_size=256, cycles=10, scm=scm)


    if "effectiveness" in args.metrics or "all" in args.metrics:
        #########################################################################################################################
        ## just test code for the produced counterfactuals -> may delete later
       # test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=7)
       # iterator = iter(test_data_loader)
      #  batch = next(iterator)
      #  counterfactuals = produce_counterfactuals(batch, scm, do_parent="digit", intervention_source=train_set)

      #  cf_image = counterfactuals["image"].squeeze(0).squeeze(0).numpy()
      #  plt.imsave("cf_img{}.png".format("thickeness"), cf_image, cmap='gray')
      #  plt.imsave("f_img.png", batch["image"].squeeze(0).squeeze(0).numpy(), cmap="gray")
        ##########################################################################################################################
        # test the predictors
        predictors = {atr: Classifier(attr=atr, width=8, num_outputs=config_cls[atr +"_num_out"], context_dim=1)
                                        if atr=="thickness"
                                        else Classifier(attr=atr, width=8, num_outputs=config_cls[atr +"_num_out"]) for atr in attribute_size.keys()}

    # load checkpoints of the predictors
        for key , cls in predictors.items():
            file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
            print(file_name)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cpu'))["state_dict"])

        #print(predictors)
        for pa in attribute_size.keys():
            evaluate_effectiveness(test_set, batch_size=256, scm=scm, attributes=list(attribute_size.keys()), do_parent=pa,
                            intervention_source=train_set, predictors=predictors)

    if "coverage_density" in args.metrics or "all" in args.metrics:
        evaluate_coverage_density(real_set=train_set, test_set=test_set, batch_size=64, scm=scm)

