"""A lot here was simply copied over from
   https://github.com/biomedia-mira/deepscm/datasets/morphomnist/__init__.py, but also extended/changed."""

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append("../../")
from datasets.morphomnist.io import load_idx

def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path

def load_morphomnist_like(root_dir, train= True, columns=None):
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)

    if columns is not None and 'index' not in columns:
        usecols = ['index'] + list(columns)
    else:
        usecols = columns

    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col='index')
    return images, labels, metrics

def normalize(data, load=False, save=True, path='./data/norm_params.pt'):
    # Normalize images
    if load:
        norm_params = torch.load(path)
        img_mean = norm_params['img_mean']
        img_std = norm_params['img_std']
        thickn_mean = norm_params['thickn_mean']
        thickn_std = norm_params['thickn_std']
        intens_mean = norm_params['intens_mean']
        intens_std = norm_params['intens_std']
    else:
        img_mean = torch.mean(data.images)
        img_std = torch.std(data.images)
        # thickness
        thickn_mean = torch.mean(data.metrics['thickness'])
        thickn_std = torch.std(data.metrics['thickness'])
        # intensity
        intens_mean = torch.mean(data.metrics['intensity'])
        intens_std = torch.std(data.metrics['intensity'])
        if save:
            norm_params = {'img_mean': img_mean, 'img_std': img_std, 'thickn_mean': thickn_mean,
                           'thickn_std': thickn_std, 'intens_mean': intens_mean, 'intens_std': intens_std}
            torch.save(norm_params, path)

    normalize = transforms.Normalize(mean=img_mean, std=img_std)
    normalized_images = transforms.Compose([normalize])(data.images)
    normalized_thickn = (data.metrics['thickness'] - thickn_mean)/thickn_std
    normalized_intens = (data.metrics['intensity'] - intens_mean)/intens_std

    return normalized_images, normalized_intens, normalized_thickn

class MorphoMNISTLike(Dataset):
    def __init__(self, attributes, train=True, columns=None, normalize_=True, transform=None, data_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')):
        self.root_dir = data_dir
        self.train = train
        self.transform = transform

        # TODO load from labels
        columns = [att for att in columns if att != 'digit']
        attributes = [att for att in attributes if att != 'digit']

        images, labels, metrics_df = load_morphomnist_like(data_dir, train, columns)
        # .copy() removes annoying warning
        self.images = torch.as_tensor(images.copy(), dtype=torch.float32)
        self.labels = torch.as_tensor(labels.copy())
        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col], dtype=torch.float32) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)
        if normalize_:
            self.images, self.metrics['intensity'], self.metrics['thickness'] = normalize(self, path=os.path.join(data_dir, 'norm_params.pt'))
        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1) for attr in attributes], dim=1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx].unsqueeze(0)
        item['attrs'] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item['attrs'])
        return item['image'], item['attrs']
