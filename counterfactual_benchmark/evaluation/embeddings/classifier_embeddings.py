import torch
import os
import sys
from json import load
sys.path.append("../../")
from models.classifiers.classifier import Classifier

class ClassifierEmbeddings():
    def __init__(self, config_file) -> None:
        assert os.path.isfile(config_file), f"{config_file} is not a file"

        with open(config_file, 'r') as f:
            config = load(f)
        attributes = [k.removesuffix('_num_out') for k in config.keys() if k.endswith('_num_out')]

        self.predictors = {
            atr: Classifier(attr=atr, num_outputs=config["attribute_size"][atr], context_dim=config[atr +"_context_dim"])
                for atr in attributes
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

    def __call__(self, x, cond, only_intensity=False, skip_attribute=None):
        intensity = cond if only_intensity else cond[:, 1].view(-1, 1)
        embeddings = []
        for name, clf in self.predictors.items():
            if name == skip_attribute:
                continue
            embeddings.append(clf.forward(x, y = intensity if name == "thickness" else None))

        return torch.cat(embeddings, dim=1)
