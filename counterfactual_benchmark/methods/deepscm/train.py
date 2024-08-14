import torch
from json import load
from importlib import import_module
import argparse
import os
from training_scripts import train_flow, train_vae, train_gan, train_diffusion

import sys
sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.adni.dataset import ADNI
from datasets.celebahq.dataset import CelebaHQ


# train_vae is used both for VAEs and HVAEs
model_to_script = {
    "flow": train_flow,
    "vae": train_vae,
    "gan": train_gan,
    "diffusion": train_diffusion
}

dataclass_mapping = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba,
    "adni": ADNI,
    "celebahq": CelebaHQ
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="./configs/celeba/simple/celeba_vae_config.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    torch.manual_seed(42)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    for variable in config["causal_graph"].keys():
        if variable not in config["mechanism_models"]:
            continue

        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])

        model = model_class(params=model_config["params"], attr_size=attribute_size)
        if "finetune" in model_config["params"] and model_config["params"]["finetune"] == 1:
            model.load_state_dict(torch.load(model_config["params"]["pretrained_path"])["state_dict"])
            model.name += '_finetuned'

        train_fn = model_to_script[config["mechanism_models"][variable]["model_type"]]
        train_fn(model,
                 config=model_config["params"],
                 data_class=dataclass_mapping[dataset],
                 graph_structure=config["causal_graph"],
                 attribute_size=attribute_size,
                 checkpoint_dir=config["checkpoint_dir"],
                 normalize_=True)