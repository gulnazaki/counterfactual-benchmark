import torch
from pytorch_lightning import Trainer
from json import load
import sys
sys.path.append("../../")
import os
import argparse
import joblib

from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.adni.dataset import ADNI
from datasets.celebahq.dataset import CelebaHQ
from datasets.transforms import get_attribute_ids
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback
from torchvision.transforms import RandomHorizontalFlip


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba,
    "celebahq": CelebaHQ,
    "adni": ADNI
}

classifier_mapping = {
    "morphomnist": Classifier,
    "celeba": CelebaClassifier,
    "celebahq": CelebaComplexClassifier,
    "adni": ADNIClassifier
}


def train_classifier(classifier, attr, train_set, val_set, config, default_root_dir, weights=None):

    ckp_callback = generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"], monitor="val_f1", mode="max")
    callbacks = [ckp_callback]

    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=[ckp_callback,
                                 generate_early_stopping_callback(patience=config["patience"], monitor="val_f1", mode="max")],
                      default_root_dir=default_root_dir, max_epochs=config["max_epochs"])

    if weights != None:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set), replacement=True)
        print("Using sampler!")
        train_data_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=config["batch_size_train"],  drop_last=False, num_workers=7)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, drop_last=False, num_workers=7)


    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)
    trainer.fit(classifier, train_data_loader, val_data_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file."
                        , default="./configs/adni_classifier_config.json")

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)

    args = parse_arguments()

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"

    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)

    dataset = config_cls["dataset"]
    attribute_size = config_cls["attribute_size"]

    attribute_ids = get_attribute_ids(attribute_size)

    if dataset == 'morphomnist':
        data = dataclass_mapping[dataset](attribute_size=attribute_size, split="train", normalize_=True)

        train_set, val_set = torch.utils.data.random_split(data, [config_cls["train_val_split"],
                                                              1-config_cls["train_val_split"]])
    else:
        # celeba, adni
        tr_transforms = RandomHorizontalFlip(0.5)
        train_set = dataclass_mapping[dataset](attribute_size=attribute_size,
                                             split="train", transform_cls=tr_transforms)
        val_set = dataclass_mapping[dataset](attribute_size=attribute_size, split="valid")

    for attribute in attribute_size.keys():
        print("Train "+ attribute +" classfier!!")
        if dataset == "adni":
            classifier = classifier_mapping[dataset](attr=attribute, num_outputs=config_cls["attribute_size"][attribute],
                                    lr=config_cls["lr"], children=config_cls["anticausal_graph"][attribute], num_slices=config_cls["attribute_size"][attribute],
                                    attribute_ids=attribute_ids, arch=config_cls["arch"])
        else:
            if dataset in {"celeba", "celebahq"}:
                if sum(attribute_size.values()) == 4:
                    classifier = CelebaComplexClassifier(attr=attribute, context_dim=len(list(config_cls["anticausal_cond"][attribute])),
                                                    num_outputs=config_cls[attribute +"_num_out"],
                                                    lr=config_cls["lr"], version=config_cls["version"])
                else:
                    classifier = CelebaClassifier(attr=attribute, num_outputs=config_cls[attribute +"_num_out"],
                                            lr=config_cls["lr"])

                if attribute == "Smiling":
                    weights = torch.tensor(joblib.load("../../datasets/celeba/weights/weights_smiling.pkl")).double()

                elif attribute == "Eyeglasses":
                    weights = torch.tensor(joblib.load("../../datasets/celeba/weights/weights_eyes.pkl")).double()

                elif attribute in {"No_Beard", "Bald"}:
                    if dataset == "celeba":
                        samples = [train_set.data[i][0] for i in range(len(train_set))]
                    else:
                        samples = [train_set.data[i] for i in range(len(train_set))]

                    labels = train_set.attrs[: , classifier.variables[attribute]].long()
                    print((labels == 1).sum(), (labels==0).sum())
                    class_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
                    print(class_count)
                    class_weights = 1. / class_count.float()

                    weights = class_weights[labels]
                    print(weights)

                elif attribute == "Bald":
                    pass

                else:
                    weights = None

            else:#morphomnist
                classifier = Classifier(attr=attribute, width=8, num_outputs=config_cls[attribute +"_num_out"],
                                        context_dim=len(list(config_cls["anticausal_cond"][attribute])), lr=config_cls["lr"])
                weights = None


        train_classifier(classifier, attribute, train_set, val_set, config_cls, default_root_dir=config_cls["ckpt_path"], weights=weights)
