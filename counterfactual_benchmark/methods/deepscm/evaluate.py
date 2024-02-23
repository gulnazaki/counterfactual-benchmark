import torch
import numpy as np
from typing import Dict, List
from json import load
from importlib import import_module
from model import SCM
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
import random

import sys
sys.path.append("../../")

from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.transforms import ReturnDictTransform

from evaluation.metrics.composition import composition
from evaluation.metrics.coverage_density import coverage_density
from evaluation.embeddings.vgg import vgg
from evaluation.metrics.effectiveness import effectiveness
from evaluation.metrics.utils import save_selected_images, save_plots
from datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from datasets.celeba.dataset import unnormalize as unnormalize_celeba

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba)
}

def produce_qualitative_samples(dataset, scm, parents, intervention_source, unnormalize_fn, num=20):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    produce_every = len(dataset) // num
    fig_idx = 0
    for i , batch in tqdm(enumerate(data_loader)):
        if i % produce_every == 0:
            res = [batch]

            for do_parent in parents:
                counterfactual = produce_counterfactuals(batch, scm, do_parent, intervention_source,
                                                         force_change=True, possible_values=dataset.possible_values)
                res.append(counterfactual)

            save_plots(res, fig_idx, parents, unnormalize_fn)
            fig_idx += 1
    return



def evaluate_coverage_density(real_set: Dataset, test_set: Dataset, batch_size: int, scm: nn.Module, attributes: List[str]):
    real_data_loader = torch.utils.data.DataLoader(real_set, batch_size=batch_size, shuffle=False, num_workers=7)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    counterfactual_images = []
    for factual_batch in tqdm(test_data_loader):
        counterfactual_batch =  produce_counterfactuals(factual_batch, scm, do_parent=random.choice(attributes), intervention_source=real_set,
                                                        force_change=True, possible_values=test_set.possible_values)
        counterfactual_images.append(counterfactual_batch['image'])

    real_images = [batch["image"] for batch in real_data_loader]
    return coverage_density(real_images, generated_images=counterfactual_images, k = 5, embedding_fn=vgg, pretrained=False)


def evaluate_composition(test_set: Dataset, unnormalize_fn, batch_size: int, cycles: List[int], scm: nn.Module, save_dir: str = "composition_samples"):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    composition_scores = []
    images = []
    for factual_batch in tqdm(test_data_loader):
        score_batch, image_batch = composition(factual_batch, unnormalize_fn, method=scm, cycles=cycles)
        composition_scores.append(score_batch)
        images.append(image_batch)

    images = np.concatenate(images)

    composition_scores = {cycle: np.concatenate([composition_batch[cycle] for composition_batch in composition_scores]) for cycle in cycles}

    os.makedirs(save_dir, exist_ok=True)
    save_selected_images(images, composition_scores[cycles[-1]], save_dir=save_dir, lower_better=True)

    for cycle in cycles:
        print(f"Average composition score for {cycle} cycles: {round(np.mean(composition_scores[cycle]), 3)}")

    return


def produce_counterfactuals(factual_batch: torch.Tensor, scm: nn.Module, do_parent:str, intervention_source: Dataset,
                            force_change: bool = False, possible_values = None, device: str = 'cuda'):
    factual_batch = {k: v.to(device) for k, v in factual_batch.items()}

    batch_size, _ , _ , _ = factual_batch["image"].shape
    idxs = torch.randperm(len(intervention_source))[:batch_size] # select random indices from train set to perform interventions

    #update with the counterfactual parent
    if force_change:
        possible_values = possible_values[do_parent]
        values = factual_batch[do_parent].cpu()
        if do_parent != "digit":
            interventions = {do_parent: torch.cat([torch.tensor(np.random.choice(possible_values[possible_values!=value])).unsqueeze(0)
                                                for value in values]).view(-1).unsqueeze(1).to(device)}
        else:
            interventions = {do_parent: torch.cat([torch.tensor(rng.choice(possible_values[torch.where((possible_values != value).any(dim=1))], axis=0)).unsqueeze(0)
                                                for value in values]).to(device)}
    else:
        interventions = {do_parent: torch.cat([intervention_source[id][do_parent] for id in idxs]).view(-1).unsqueeze(1).to(device)
                        if do_parent!="digit" else torch.cat([intervention_source[id][do_parent].unsqueeze(0).to(device) for id in idxs])}

    abducted_noise = scm.encode(**factual_batch)
    counterfactual_batch = scm.decode(interventions, **abducted_noise)

    return counterfactual_batch


def evaluate_effectiveness(test_set: Dataset, unnormalize_fn, batch_size:int , scm: nn.Module, attributes: List[str], do_parent:str,
                           intervention_source: Dataset, predictors: Dict[str, Classifier]):

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    effectiveness_scores = {attr_key: [] for attr_key in attributes}
    for factual_batch in tqdm(test_data_loader):
        counterfactuals = produce_counterfactuals(factual_batch, scm, do_parent, intervention_source,
                                                  force_change=True, possible_values=test_set.possible_values)
        e_score = effectiveness(counterfactuals, unnormalize_fn, predictors)

        for attr in attributes:
            effectiveness_scores[attr].append(e_score[attr])

    effectiveness_score = {key  : round(np.mean(score), 3) for key, score in effectiveness_scores.items()}

    print(f"Effectiveness score do({do_parent}): {effectiveness_score}")

    return effectiveness_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/celeba_vae_config.json")
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file.", default="./configs/celeba_classifier_config.json")
    parser.add_argument("--metrics", '-m',
                        nargs="+", type=str,
                        help="Metrics to calculate. "
                        "Choose one or more of [composition, effectiveness, coverage_density] or use 'all'.",
                        default=["all"])
    parser.add_argument("--cycles", '-cc', nargs="+", type=int, help="Composition cycles.", default=[1, 10])
    parser.add_argument("--coverage-density-on-train", '-cvtrain', action='store_true', help="Whether to compute coverage & density against the training set")
    parser.add_argument("--qualitative", '-qn', type=int, help="Number of qualitative results to produce", default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
   # torch.manual_seed(42)

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"
    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    models = {}
    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(params=model_config["params"], attr_size=attribute_size)

        models[variable] = model

    batch_size = config["mechanism_models"]["image"]["params"]["batch_size_val"]

    scm = SCM(checkpoint_dir=config["checkpoint_dir"],
              graph_structure=config["causal_graph"],
              **models)

    dataset = config["dataset"]
    data_class, unnormalize_fn = dataclass_mapping[dataset]

    transform = ReturnDictTransform(attribute_size)

    train_set = data_class(attribute_size, split='train', transform=transform)
    test_set = data_class(attribute_size, split='test', transform=transform)

    if args.qualitative > 0:
        produce_qualitative_samples(dataset=test_set, scm=scm, parents=list(attribute_size.keys()),
                                    intervention_source=train_set, unnormalize_fn=unnormalize_fn, num=args.qualitative)


    if "composition" in args.metrics or "all" in args.metrics:
        evaluate_composition(test_set, unnormalize_fn, batch_size, cycles=args.cycles, scm=scm)


    if "effectiveness" in args.metrics or "all" in args.metrics:
        if dataset == "morphomnist":
            predictors = {atr: Classifier(attr=atr, width=8, num_outputs=config_cls[atr +"_num_out"], context_dim=1)
                                        if atr=="thickness"
                                        else Classifier(attr=atr, width=8, num_outputs=config_cls[atr +"_num_out"]) for atr in attribute_size.keys()}


        else:
             predictors = {atr: CelebaClassifier(attr=atr, width=64,
                                          num_outputs=config_cls[atr +"_num_out"], lr=config_cls["lr"]) for atr in attribute_size.keys()}


        # load checkpoints of the predictors
        for key , cls in predictors.items():
            file_name = next((file for file in os.listdir(config_cls["ckpt_path"]) if file.startswith(key)), None)
            print(file_name)
            cls.load_state_dict(torch.load(config_cls["ckpt_path"] + file_name , map_location=torch.device('cuda'))["state_dict"])
            cls.to('cuda')

        #print(predictors)
        for pa in attribute_size.keys():
            evaluate_effectiveness(test_set, unnormalize_fn, batch_size, scm=scm, attributes=list(attribute_size.keys()), do_parent=pa,
                            intervention_source=train_set, predictors=predictors)

    if "coverage_density" in args.metrics or "all" in args.metrics:
        real_set = train_set if args.coverage_density_on_train else test_set
        evaluate_coverage_density(real_set, test_set=test_set, batch_size=64, scm=scm, attributes=list(attribute_size.keys()))

