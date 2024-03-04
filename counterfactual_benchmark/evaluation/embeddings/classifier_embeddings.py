import torch
import os
import sys
from json import load
sys.path.append("../../")
from models.classifiers.classifier import Classifier

class ClassifierEmbeddings():
    def __init__(self, config_file) -> None:
        assert os.path.isfile(config_file), f"{config_file} is not a file"
        # assert config_file.startswith('morphomnist_'), "Only MorphoMNIST classifiers supported"

        with open(config_file, 'r') as f:
            config = load(f)
        attributes = [k.removesuffix('_num_out') for k in config.keys() if k.endswith('_num_out')]

        self.predictors = {
            atr: Classifier(attr=atr, width=8, num_outputs=config[atr + "_num_out"], context_dim=1) if atr == "thickness"
                else Classifier(attr=atr, width=8, num_outputs=config[atr + "_num_out"]) for atr in attributes
        }

        # load checkpoints of the predictors
        ckpt_path = config["ckpt_path"]
        if not os.path.isdir(ckpt_path):
            print(f"Provided ckpt_path {ckpt_path} is not a directory")

        for key, clf in self.predictors.items():
            file_name = next((file for file in os.listdir(ckpt_path) if file.startswith(key)), None)
            print(file_name)
            clf.load_state_dict(torch.load(os.path.join(ckpt_path,file_name), map_location=torch.device('cuda'))["state_dict"])
            clf.fc = torch.nn.Sequential(*[clf.fc[i] for i in range(3)])
            for param in clf.parameters():
                param.requires_grad = False
            clf.eval()
            clf.to('cuda')

    def __call__(self, x, cond, only_intensity=False):
        intensity = cond if only_intensity else cond[:, 1].view(-1, 1)
        embeddings = []
        for name, clf in self.predictors.items():
            embeddings.append(clf.forward(x, y = intensity if name == "thickness" else None))

        return torch.cat(embeddings, dim=1)