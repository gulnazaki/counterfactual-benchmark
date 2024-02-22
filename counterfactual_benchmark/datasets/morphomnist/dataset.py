"""A lot here was simply copied over from
   https://github.com/biomedia-mira/deepscm/datasets/morphomnist/__init__.py, but also extended/changed."""

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import sys
sys.path.append("../../")
from datasets.morphomnist.io import load_idx

MIN_MAX = {
    "thickness": [0.87598526, 6.255515],
    "intensity": [66.601204, 254.90317],
    "image": [0.0, 255.0]
}

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
    # # Normalize images
    # if load:
    #     norm_params = torch.load(path)
    #     img_mean = norm_params['img_mean']
    #     img_std = norm_params['img_std']
    #     thickn_mean = norm_params['thickn_mean']
    #     thickn_std = norm_params['thickn_std']
    #     intens_mean = norm_params['intens_mean']
    #     intens_std = norm_params['intens_std']
    # else:
    #     img_mean = torch.mean(data.images)
    #     img_std = torch.std(data.images)
    #     # thickness
    #     thickn_mean = torch.mean(data.metrics['thickness'])
    #     thickn_std = torch.std(data.metrics['thickness'])
    #     # intensity
    #     intens_mean = torch.mean(data.metrics['intensity'])
    #     intens_std = torch.std(data.metrics['intensity'])
    #     if save:
    #         norm_params = {'img_mean': img_mean, 'img_std': img_std, 'thickn_mean': thickn_mean,
    #                        'thickn_std': thickn_std, 'intens_mean': intens_mean, 'intens_std': intens_std}
    #         torch.save(norm_params, path)

    # normalize = transforms.Normalize(mean=img_mean, std=img_std)
    # normalized_images = transforms.Compose([normalize])(data.images)

    # normalized_thickn = (data.metrics['thickness'] - thickn_mean)/thickn_std
    # normalized_intens = (data.metrics['intensity'] - intens_mean)/intens_std

    # return normalized_images, normalized_intens, normalized_thickn
    normalized = {}
    for k, v in MIN_MAX.items():
        value = data.images if k == "image" else data.metrics[k]
        # [min,max] -> [0,1]
        normalized[k] = (value - v[0]) / (v[1] - v[0])
        # [0,1] -> [-1,1]
        normalized[k] = 2 * normalized[k] - 1

    return normalized["image"], normalized["intensity"], normalized["thickness"]

def unnormalize(value, name):
    # [-1,1] -> [0,1]
    value = (value + 1) / 2
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

class MorphoMNISTLike(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True, transform=None, data_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')):
        self.has_valid_set = False
        self.root_dir = data_dir
        self.train = True if split == 'train' else False
        self.transform = transform
        self.pad = transforms.Pad(padding=2)

        # digit is loaded from labels
        columns = [att for att in attribute_size.keys() if att != 'digit']

        images, labels, metrics_df = load_morphomnist_like(data_dir, self.train, columns)
        # .copy() removes annoying warning
        self.images = self.pad(torch.as_tensor(images.copy(), dtype=torch.float32))
        self.labels = F.one_hot(torch.as_tensor(labels.copy(), dtype=torch.long), num_classes=10)

        if columns is None:
            columns = metrics_df.columns
        self.metrics = {col: torch.as_tensor(metrics_df[col], dtype=torch.float32) for col in columns}
        self.columns = columns
        assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)
        if normalize_:
            self.images, self.metrics['intensity'], self.metrics['thickness'] = normalize(self, path=os.path.join(data_dir, 'norm_params.pt'))

        if "digit" in attribute_size.keys():
            self.metrics["digit"] = self.labels

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1) if attr != "digit" else self.metrics[attr]
                                for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx].unsqueeze(0)
        item['attrs'] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item['attrs'])
        return item['image'], item['attrs']
