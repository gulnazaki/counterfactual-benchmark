from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype
import torch

MIN_MAX = [0.0, 255.0]

def load_data(data_dir, train=True):
    transforms = Compose([CenterCrop(150), Resize((128, 128)), ToTensor(), ConvertImageDtype(dtype=torch.float32),])
    data = CelebA(root=data_dir, split='train' if train else 'test', transform=transforms, download=False)
    return data

def normalize(data):
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

class Celeba(Dataset):
    def __init__(self, attribute_size, train=True, normalize_=True, transform=None, data_dir='/storage/th.melistas/'):
        super().__init__()
        self.attribute_sizes = attribute_size
        self.transform = transform
        self.data = load_data(data_dir, train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx][0])
        # if self.as_dict:
        #     return {**{attr : self.cont_attr[idx][vars2int[attr]].view(1, -1) for attr in attrs}, "image" : self.data[idx][0].unsqueeze(0)}
        return self.data[idx]

