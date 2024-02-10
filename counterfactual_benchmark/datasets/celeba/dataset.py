import torch
import torchvision
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F

import sys
sys.path.append("../../")

class CelebALike(Dataset):
    def __init__(self, split = "train", transform = None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.data = datasets.CelebA(root="/storage/n.spyroudata", split = self.split, 
                                                 target_type = "attr", transform=self.transform, download = True)

    
    def __getitem__(self, index):
        item = self.data[index]
        
        return item
    

if __name__ == "__main__":
    tr_dataset = CelebALike(split="train")
    a = tr_dataset[0]
    print(a)