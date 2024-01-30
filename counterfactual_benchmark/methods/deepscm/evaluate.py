import torch
import numpy as np
from json import load
from importlib import import_module
from model import SCM

import sys
sys.path.append("../../")
from datasets.morphomnist.dataset import MorphoMNISTLike
from evaluation.metrics.composition import composition

from datasets.transforms import ReturnLabelsTransform


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike
}


def evaluate(test_set, batch_size, scm, attributes):
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=7)

    # composition
    composition_scores = []
    for factual_batch in test_data_loader:
        composition_scores.append(composition(factual_batch, method=scm, cycles=10))
    composition_score = np.mean(composition_scores)
    print(composition_score)

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
    test_set = data_class(attributes=attributes, train=False, columns=attributes, transform=transform)

    evaluate(test_set, batch_size=256, scm=scm, attributes=attributes)

