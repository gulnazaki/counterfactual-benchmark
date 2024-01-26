import torch
from pytorch_lightning import Trainer

import sys
sys.path.append("../../")
from datasets.transforms import SelectAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback

def train_flow(flow, config, data_class, graph_structure, attrs, checkpoint_dir, **kwargs):
    transform = SelectAttributesTransform(attrs.index(flow.name), [attrs.index(attr_pa) for attr_pa in graph_structure[flow.name]])
    # load the data (with continuous labels)
    data = data_class(attributes=attrs, transform=transform, **kwargs)
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", callbacks=[generate_checkpoint_callback(flow.name + "_flow", checkpoint_dir),
                                                                                      generate_early_stopping_callback(patience=config["patience"])],
                                                                                                                       default_root_dir=checkpoint_dir,
                                                                                                                       max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(flow, train_data_loader, val_data_loader)


def train_vae(vae_img, train_set, val_set, checkpoint_dir, config):
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", gradient_clip_val=config["gradient_clip_val"], callbacks=[generate_checkpoint_callback(config["name"], config["checkpoint_dir"]),
                                 generate_early_stopping_callback(patience=config["patience"])], checkpoint_dir=checkpoint_dir, max_epochs=config["max_epochs"])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False)
    trainer.fit(vae_img, train_data_loader, val_data_loader)