import torch
from json import load
from importlib import import_module
from training_scripts import train_flow, train_vae

import sys
sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike


model_to_script = {
    "flow": train_flow,
    "vae": train_vae
}

dataclass_mapping = {
    "morphomnist": MorphoMNISTLike
}


if __name__ == "__main__":
    torch.manual_seed(42)
    config_file = "configs/morphomnist_config.json"
    with open(config_file, 'r') as f:
        config = load(f)

    dataset = config["dataset"]
    attributes = config["causal_graph"]["image"]

    for variable in config["causal_graph"].keys():
        model_config = config["mechanism_models"][variable]

        module = import_module(model_config["module"])
        model_class = getattr(module, model_config["model_class"])
        model = model_class(name=variable, params=model_config["params"], attrs=attributes)

        train_fn = model_to_script[config["mechanism_models"][variable]["model_type"]]
        train_fn(model,
                 config=model_config["params"],
                 data_class=dataclass_mapping[dataset],
                 graph_structure=config["causal_graph"],
                 attrs=attributes,
                 checkpoint_dir=config["checkpoint_dir"],
                 columns=attributes,
                 normalize_=True,
                 train=True)

