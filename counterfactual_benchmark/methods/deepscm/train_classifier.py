import torch
from pytorch_lightning import Trainer
from json import load
import sys
sys.path.append("../../")
import sys

from datasets.morphomnist.dataset import MorphoMNISTLike
from models.classifiers.classifier import Classifier
#from datasets.transforms import SelectAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback


def train_classifier(classifier, attr, train_set, val_set, config, default_root_dir):

    ckp_callback = generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"])

   # if attr == "digit":
   #     ckp_callback = generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"], monitor=None)

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", 
                      callbacks=[ckp_callback, 
                                 generate_early_stopping_callback(patience=config["patience"])], 
                      default_root_dir=default_root_dir, max_epochs=config["max_epochs"])
    
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(classifier, train_data_loader, val_data_loader)


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike
}


if __name__ == "__main__":
    torch.manual_seed(42)
    config_file = "configs/morphomnist_config.json"
    config_file_cls = "configs/morphomnist_classifier_config.json"

    with open(config_file, 'r') as f:
        config = load(f)
    
    with open(config_file_cls, 'r') as f1:
        config_cls = load(f1)

    dataset = config["dataset"]
   # attributes = config["causal_graph"]["image"]
    attribute_size = config["attribute_size"]
  

    data = dataclass_mapping[dataset](attribute_size=attribute_size, normalize_=True, train=True)

    train_set, val_set = torch.utils.data.random_split(data, [config_cls["train_val_split"], 
                                                              1-config_cls["train_val_split"]])
    
  
    for attribute in ["digit"]:#attribute_size.keys():
        print("Train "+ attribute +" classfier!!")
        if attribute == "thickness":
            classifier = Classifier(attr=attribute, width=8, num_outputs=config_cls[attribute +"_num_out"],
                                     context_dim=1, lr=config_cls["lr"])
        
        else:
            classifier = Classifier(attr=attribute, width=8, 
                                    num_outputs=config_cls[attribute +"_num_out"], lr=config_cls["lr"])
            
        
            
        train_classifier(classifier, attribute, train_set, val_set, config_cls, default_root_dir=config_cls["ckpt_path"])

        

    


    









