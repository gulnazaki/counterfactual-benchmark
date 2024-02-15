import torch
from pytorch_lightning import Trainer

import sys
sys.path.append("../../")
from datasets.transforms import SelectAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback

def train_flow(flow, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    transform = SelectAttributesTransform(flow.name, attribute_size, graph_structure)
    # load the data (with continuous labels)
    data = data_class(attribute_size=attribute_size, transform=transform, **kwargs)
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", callbacks=[generate_checkpoint_callback(flow.name + "_flow", checkpoint_dir),
                                                                                      generate_early_stopping_callback(patience=config["patience"])],
                                                                           default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(flow, train_data_loader, val_data_loader)


def train_vae(vae, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    data = data_class(attribute_size=attribute_size, **kwargs)

    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)

    callbacks = [
        generate_checkpoint_callback(vae.name, checkpoint_dir),
        generate_early_stopping_callback(patience=config["patience"])
    ]

    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", gradient_clip_val=config["gradient_clip_val"],
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(vae, train_data_loader, val_data_loader)
    
    
    
    
    
def train_gan(gan, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    data = data_class(attribute_size=attribute_size, **kwargs)
  
    # split into train and validation
    train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)



    callbacks = [
        generate_checkpoint_callback(gan.name, checkpoint_dir),
        generate_early_stopping_callback(patience=config["patience"])
    ]


    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    if config['finetune'] == 0:
        trainer.fit(gan, train_data_loader, val_data_loader)
    else:
        trainer.fit(gan, train_data_loader, val_data_loader,  ckpt_path ='')