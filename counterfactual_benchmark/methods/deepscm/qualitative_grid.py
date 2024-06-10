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
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")

from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.celebahq.dataset import CelebaHQ
from datasets.adni.dataset import ADNI
from datasets.transforms import ReturnDictTransform
from datasets.morphomnist.dataset import unnormalize as unnormalize_morphomnist
from datasets.celeba.dataset import unnormalize as unnormalize_celeba
from datasets.adni.dataset import unnormalize as unnormalize_adni
from methods.deepscm.evaluate import produce_counterfactuals
from evaluation.metrics.utils import to_value

torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng()

dataclass_mapping = {
    "morphomnist": (MorphoMNISTLike, unnormalize_morphomnist),
    "celeba": (Celeba, unnormalize_celeba),
    "adni": (ADNI, unnormalize_adni),
    "celebahq": (CelebaHQ, unnormalize_celeba)
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="Config file for experiment.", default="./configs/qualitative_grid_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    torch.manual_seed(42)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        grid_config = load(f)

    datasets = list(grid_config.keys())

    num_of_grids = 2
    save_dir = './qualitative_grid'

    grid = {}

    for dataset_name in datasets:
        dataset = grid_config[dataset_name]
        var = dataset["variable"]
        grid[dataset_name] = {"factuals": []}
        grid[dataset_name]["variable"] = var
        for config_name in grid_config[dataset_name]["configs"]:
            config_file = os.path.join("configs", config_name)
            assert os.path.isfile(config_file), f"{config_file} is not a file"
            with open(config_file, 'r') as f:
                config = load(f)

            models = {}
            for variable in config["causal_graph"].keys():
                if variable not in config["mechanism_models"]:
                    continue
                model_config = config["mechanism_models"][variable]

                module = import_module(model_config["module"])
                model_class = getattr(module, model_config["model_class"])
                model = model_class(params=model_config["params"], attr_size=config["attribute_size"])

                models[variable] = model
                if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
                    model.name += '_finetuned'

            scm = SCM(checkpoint_dir=config["checkpoint_dir"],
                    graph_structure=config["causal_graph"],
                    temperature=0.1,
                    **models)

            dataset = config["dataset"]
            data_class, unnormalize_fn = dataclass_mapping[dataset]
            grid[dataset_name]['unnormalize'] = unnormalize_fn

            transform = ReturnDictTransform(config["attribute_size"])

            test_set = data_class(config["attribute_size"], split='test', transform=transform)
            data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)


            model_type = config["mechanism_models"]["image"]["model_type"]
            if config["mechanism_models"]["image"]["model_class"].endswith("HVAE"):
                model_type = 'hvae'

            grid[dataset_name][model_type] = []
            data = iter(data_loader)
            for i in range(num_of_grids):
                factual = next(data)
                np.random.seed(42)
                counterfactual = produce_counterfactuals(factual, scm, var, intervention_source=None,
                                                            force_change=True, possible_values=test_set.possible_values)

                grid[dataset_name]["factuals"].append(factual)
                grid[dataset_name][model_type].append(counterfactual)


    models = ["vae", "hvae", "gan"]
    fig, axs = plt.subplots(len(grid), len(models) + 1)
    horizontal_titles = ["factuals"] + models

    for i, dataset in enumerate(grid.keys()):
        variable = grid[dataset]["variable"]
        unnormalize_fn = grid[dataset]['unnormalize']

        for j, image_type in enumerate(horizontal_titles):
            image = grid[dataset][image_type][0]["image"]
            img = unnormalize_fn(image.cpu().squeeze(0), name="image")
            if img.shape[0] == 3:
                axs[i][j].imshow(img.permute(1, 2, 0))
            else:
                axs[i][j].imshow(img[0], cmap='gray')

            if i == 0:
                axs[i][j].set_title(image_type.upper() if image_type in models else image_type)
            if j == 0:
                # axs[i][j].set(ylabel=dataset.replace('_', ' ').title() + '\n' + f'do({variable}) = {to_value(grid[dataset][image_type][0][variable], variable)})')
                axs[i][j].set(ylabel=f'do({variable})\n{to_value(grid[dataset][models[0]][0][variable], variable, unnormalize_fn)}')

            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"grid_{0}.png"))
    plt.close()
