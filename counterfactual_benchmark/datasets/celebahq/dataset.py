from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype
import torch
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

MIN_MAX = {
    'image': [0.0, 255.0]
}


# def normalize(value, name):
#     # already in [0,1]
#     # [0,1] -> [-1,1]
#     value = 2 * value - 1
#     return value

def unnormalize(value, name):
    # [-1,1] -> [0,1]
    # value = (value + 1) / 2
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value.to(torch.uint8)


def load_data(root_dir="/home/ubuntu/CelebAMask-HQ", split= "train"):

    attributes = pd.read_csv(os.path.join(root_dir, "CelebAMask-HQ-attribute-anno.txt"),
                                      delimiter=" ", skiprows=1)
    attributes[attributes==-1] = 0
    mapping = pd.read_csv(os.path.join(root_dir, "CelebA-HQ-to-CelebA-mapping.txt"),
                                      delimiter=" ", skipinitialspace=True)

    splits = pd.read_csv(os.path.join(root_dir, "list_eval_partition.txt"),
                                      delimiter=" ", header=None)

    idxs , orig_idxs = mapping["idx"].to_list(), mapping["orig_idx"].to_list()
    mapping_dict = {key: value for key, value in zip(idxs, orig_idxs)}

    orig_indexes , num_split = splits[0].to_list(), splits[1].to_list()
    data_to_split = {int(key.split(".")[0]): value for key, value in zip(orig_indexes, num_split)}

    transform = Compose([Resize((256, 256)), ToTensor(), ConvertImageDtype(dtype=torch.float32)])
    split_map = {'train': 0, 'valid': 1, 'test': 2}

    all_images_names = sorted(os.listdir(os.path.join(root_dir, "CelebA-HQ-img")),
                                       key=lambda x : int(x.split(".")[0]))
    data = []
    selected_indices = []
    for file in tqdm(all_images_names):
        if data_to_split[mapping_dict[int(file.split(".")[0])]] == split_map[split]:
                img = Image.open(os.path.join(root_dir, "CelebA-HQ-img/" + file)).convert("RGB")
                data.append(transform(img))
                selected_indices.append(int(file.split(".")[0]))


    attributes = attributes.iloc[list(selected_indices)]

    return data, attributes




class CelebaHQ(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True,
                 transform=None, transform_cls=None, data_dir='/storage/n.spyrou/CelebAMask-HQ'):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        self.data, self.attributes = load_data(data_dir, split)

        #attribute_ids = [self.attributes.columns.get_loc(attr) for attr in attribute_size.keys()]
        self.metrics = {attr: torch.as_tensor(list(self.attributes[attr]), dtype=torch.float32) for attr in attribute_size.keys()}

       # self.metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.float32) for attr, attr_id in zip(attribute_size.keys(), attribute_ids)}

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1)
                                for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}
        self.bins = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx], self.attrs[idx])

        if self.transform_cls:
            return self.transform_cls(self.data[idx]), self.attrs[idx]

        return self.data[idx], self.attrs[idx]






if __name__ == "__main__":
    #data, attributes = load_data(split="test")
   # print(len(data))

    attribute_size = {
        "Young": 1,
        "Male": 1,
        "No_Beard": 1,
        "Bald" : 1
     }

    dataset = CelebaHQ(attribute_size=attribute_size, split="test")


    print(len(dataset))
    a = dataset[0]
    print(a[0].shape, a[1].shape, a[1])

    #print(data.attributes)
   # attrs = data.attributes
    #total = len(attrs)


    #print(attrs)


