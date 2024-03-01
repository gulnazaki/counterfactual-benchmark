import torch
from pytorch_lightning import Trainer
import sys
sys.path.append("../../")
from datasets.transforms import SelectParentAttributesTransform
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback


def get_dataloaders(data_class, attribute_size, config, transform=None, **kwargs):
    data = data_class(attribute_size=attribute_size, transform=transform, split='train', **kwargs)

    if data.has_valid_set:
        train_set = data
        val_set = data_class(attribute_size=attribute_size, split='valid', **kwargs)
    else:
        train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])

    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)
    return train_data_loader, val_data_loader


def train_flow(flow, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    transform = SelectParentAttributesTransform(flow.name.rstrip('_flow'), attribute_size, graph_structure)

    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, transform, **kwargs)
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", callbacks=[generate_checkpoint_callback(flow.name, checkpoint_dir),
                                                                                      generate_early_stopping_callback(patience=config["patience"])],
                                                                           default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(flow, train_data_loader, val_data_loader)


def train_vae(vae, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):
    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, **kwargs)

    callbacks = [
        generate_checkpoint_callback(vae.name, checkpoint_dir),
        generate_early_stopping_callback(patience=config["patience"])
    ]
    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(vae, train_data_loader, val_data_loader)


def train_gan(gan, config, data_class, graph_structure, attribute_size, checkpoint_dir, **kwargs):

    train_data_loader, val_data_loader = get_dataloaders(data_class, attribute_size, config, **kwargs)

    # split into train and validation
    #train_set, val_set = torch.utils.data.random_split(data, [config["train_val_split"], 1 - config["train_val_split"]])
    #train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=7)
    #val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=7)


    if config['finetune'] == 0:
        min_delta = 0.01
        monitor="fid"
    else:
        min_delta = 0.001
        monitor="lpips"
    callbacks = [
        generate_checkpoint_callback(gan.name, checkpoint_dir, monitor=monitor),
        generate_early_stopping_callback(patience=config["patience"], min_delta=min_delta, monitor=monitor)
    ]


    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=callbacks,
                      default_root_dir=checkpoint_dir, max_epochs=config["max_epochs"])

    trainer.fit(gan, train_data_loader, val_data_loader)
