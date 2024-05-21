import torch
from pytorch_lightning import Trainer
from json import load
import sys
sys.path.append("../../")
import sys, os
import argparse
import joblib 

from datasets.morphomnist.dataset import MorphoMNISTLike
from datasets.celeba.dataset import Celeba
from datasets.celebahq.dataset import CelebaHQ
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback
from torchvision.transforms import Compose, AutoAugment, RandomHorizontalFlip



def train_classifier(classifier, attr, train_set, val_set, config, default_root_dir, weights = None):

    ckp_callback = generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"], monitor="val_loss")
    callbacks = [ckp_callback]

    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=[ckp_callback,
                                 generate_early_stopping_callback(patience=config["patience"], monitor="val_loss")],
                      default_root_dir=default_root_dir, max_epochs=config["max_epochs"])

    if weights!=None:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set), replacement=True)
        print("USE SAMPLER!!!")
        train_data_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=config["batch_size_train"],  drop_last=False)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"],  shuffle=True, drop_last=False)


    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(classifier, train_data_loader, val_data_loader)


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba,
    "celebahq": CelebaHQ
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file."
                        , default="./configs/celeba_complex_classifier.json")

    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)

    args = parse_arguments()

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"

    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)


    dataset = config_cls["dataset"]
    attribute_size = config_cls["attribute_size"]


    if dataset in {"celeba", "celebahq"}: #celeba or celebahq
        tr_transforms = Compose([RandomHorizontalFlip()])
        train_set = dataclass_mapping[dataset](attribute_size=attribute_size, 
                                             split="train", transform_cls=tr_transforms)
        
        val_set = dataclass_mapping[dataset](attribute_size=attribute_size, split="valid")
    
    else:
        data = dataclass_mapping[dataset](attribute_size=attribute_size, split="train", normalize_=True)

        train_set, val_set = torch.utils.data.random_split(data, [config_cls["train_val_split"],
                                                              1-config_cls["train_val_split"]])

    for attribute in attribute_size.keys():
        print("Train "+ attribute +" classfier!!")

        if dataset in {"celeba", "celebahq"}:

            if sum(attribute_size.values()) == 4:
            
                classifier = CelebaComplexClassifier(attr=attribute, context_dim=len(list(config_cls["anticausal_cond"][attribute])), 
                                                  num_outputs=config_cls[attribute +"_num_out"], 
                                                  lr=config_cls["lr"], version=config_cls["version"])
                

        #simple celeba graph: Smiling, Eyeglasses
            else:
                classifier = CelebaClassifier(attr=attribute, num_outputs=config_cls[attribute +"_num_out"], 
                                          lr=config_cls["lr"])
            
            if attribute == "Smiling":
                weights = torch.tensor(joblib.load("../../datasets/celeba/weights/weights_smiling.pkl")).double()
        
            elif attribute == "Eyeglasses":
                weights = torch.tensor(joblib.load("../../datasets/celeba/weights/weights_eyes.pkl")).double()
            else:
                weights = None
                

        else:#morphomnist
            classifier = Classifier(attr=attribute, width=8, num_outputs=config_cls[attribute +"_num_out"],
                                     context_dim=len(list(config_cls["anticausal_cond"])), lr=config_cls["lr"])


        train_classifier(classifier, attribute, train_set, val_set, config_cls, default_root_dir=config_cls["ckpt_path"], weights=weights)