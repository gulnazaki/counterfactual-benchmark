import torch
from json import load
from importlib import import_module
from training_scripts import train_flow, train_vae, train_gan

import sys

sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba

model_to_script = {
    "flow": train_flow,
    "gan": train_gan
}

dataclass_mapping = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba
}

if __name__ == "__main__":
    args = parse_arguments()
    torch.manual_seed(42)

    assert os.path.isfile(args.config), f"{args.config} is not a file"
    with open(args.config, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attribute_size = config["attribute_size"]

    for variable in config["causal_graph"].keys():
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(name=variable, params=model_config["params"], attr_size=attribute_size)

        train_fn = model_to_script[config["mechanism_models"][variable]["model_type"]]

        train_fn(model,
                 config=model_config["params"],
                 data_class=dataclass_mapping[dataset],
                 graph_structure=config["causal_graph"],
                 attribute_size=attribute_size,
                 checkpoint_dir=config["checkpoint_dir"],
                 normalize_=True,
                 train=True)

