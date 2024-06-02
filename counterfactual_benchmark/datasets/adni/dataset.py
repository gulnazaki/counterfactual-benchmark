import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms

MIN_MAX = {
    'image': [0.0, 255.0],
    'age': [55.1, 89.3],
    'brain_vol': [669364.0, 1350180.0],
    'vent_vol': [5834.0, 145115.0]
}

ATTRIBUTE_MAPPING = {
    "apoE": "APOE4",
    # "av45": "AV45",
    # "pT": "PTAU",
    "age": "AGE",
    "sex": "PTGENDER",
    "brain_vol": "WholeBrain",
    "vent_vol": "Ventricles",
    # "education": "PTEDUCAT",
    # "moc_score": "MOCA"
}

TOTAL_SLICES = 30

SPLIT_TRAIN_VAL_TEST = [
    ('train', 0.7),
    ('valid', 0.15),
    ('test', 0.15)
]

def split_subjects(subjects, split):
    offset = 0
    for split_, percent in SPLIT_TRAIN_VAL_TEST:
        split_size = int(percent * len(subjects))
        if split_ == split:
            return subjects[offset:offset+split_size]
        offset += split_size
    return subjects[offset:offset+split_size]

def bin_array(num, m=None, reverse=False):
    if reverse:
        return torch.matmul(num, torch.flip(torch.arange(num.shape[1], dtype=torch.float32), [0]).to(num.device))
    else:
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32)

def ordinal_array(num, m=None, reverse=False, scale=1):
    if reverse:
        return scale * torch.count_nonzero(num, dim=1).to(num.device)
    else:
        return np.pad(np.ones(num), (m - num, 0), 'constant').astype(np.float32)

def encode_attribute(value, name, num_of_slices=30):
    if name == 'sex':
        return 0 if value == 'Female' else 1
    elif name == "apoE":
        return bin_array(int(value), 2)
    elif name == "slice":
        return ordinal_array(int(value), num_of_slices)
    else:
        return float(value)

def fix_age(initial_age, visit):
    return initial_age if visit == 'bl' else initial_age + float(visit[1:]) / 12

def normalize(value, name):
    if name not in MIN_MAX:
        return value
    # [min,max] -> [0,1]
    value = (value - MIN_MAX[name][0]) / (MIN_MAX[name][1] - MIN_MAX[name][0])
    return value

def unnormalize(value, name):
    if name not in MIN_MAX:
        return value
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

def load_data(data_dir, normalize_=True, num_of_slices=30, split='train', keep_only_screening=False):
    # TODO also try [-1,1]
    def img_to_0_1(img):
        # [-1,1] -> [0,1]
        return (img + 1) / 2 if normalize_ else img

    leave_out_first = (TOTAL_SLICES - num_of_slices) // 2
    valid_range = range(leave_out_first, leave_out_first + num_of_slices)

    attributes = {'subject': [],
                'slice': [],
                'date': []}
    subject_dates_dict = {}
    images = []
    subject_paths = split_subjects(sorted(Path(data_dir).glob('*')), split)
    for subject_path in subject_paths:
        dates = sorted(list(subject_path.glob('*/*')), key=lambda d: d.name)
        if keep_only_screening:
            dates = dates[:1]
            # these subjects have volume measures for the second date
            if subject_path.name in ["123_S_0050", "137_S_0825"]:
                dates = dates[1:2]
        subject_dates_dict[subject_path.name] = []
        for date_path in dates:
            date = date_path.name
            subject_dates_dict[subject_path.name].append(date)

            for image_path in sorted(date_path.glob('*/*.tiff')):
                slice = int(image_path.stem.split('slice')[1])
                if slice in valid_range:
                    images.append(img_to_0_1(np.array(Image.open(image_path))))
                    attributes['subject'].append(subject_path.name)
                    attributes['slice'].append(encode_attribute(slice - leave_out_first, 'slice', num_of_slices))
                    attributes['date'].append(date)

    return np.array(images), attributes, subject_dates_dict

def load_extra_attributes(csv_path, attributes, attribute_dict, subject_dates_dict, keep_only_screening=False):
    index_col = 'PTID'
    usecols = [index_col, 'VISCODE', 'EXAMDATE'] + [ATTRIBUTE_MAPPING[att] for att in attributes if att in ATTRIBUTE_MAPPING]
    df = pd.read_csv(csv_path, usecols=usecols, index_col=index_col).sort_index()

    for att in ATTRIBUTE_MAPPING:
        if att in attributes:
            attribute_dict[att] = []

    indices_to_remove = []

    for idx, (subject, date) in enumerate(zip(attribute_dict['subject'], attribute_dict['date'])):
        subject_df = df.loc[subject].sort_values(by='EXAMDATE')

        # these subjects have volume measures for the second exam
        if keep_only_screening and subject in ["123_S_0050", "137_S_0825"]:
            date_idx = 1
        else:
            date_idx = subject_dates_dict[subject].index(date)

        if date_idx >= len(subject_df) or subject_df.iloc[date_idx].isnull().any().any():
            # print(f'Skipping {subject} {date}')
            indices_to_remove.append(idx)
            continue

        for att, csv_att in ATTRIBUTE_MAPPING.items():
            if att in attributes:
                value = encode_attribute(subject_df[csv_att].iloc[date_idx], att)
                if att == 'age':
                    value = fix_age(initial_age=value, visit=subject_df["VISCODE"].iloc[date_idx])
                attribute_dict[att].append(value)

    del attribute_dict['subject']
    del attribute_dict['date']
    return attribute_dict, indices_to_remove


class ADNI(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True,
                 transform=None, transform_cls=None, num_of_slices=30, keep_only_screening=False,
                 data_dir='/home/ubuntu/ADNI/preprocessed_data',
                 csv_path='/home/ubuntu/ADNI/ADNIMERGE_22Apr2024.csv'):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        # pad: 180x180 -> 192x192
        self.pad = transforms.Pad(padding=6)

        num_of_slices = attribute_size['slice']

        assert num_of_slices <= 30, "The 30 middle slices have been saved"
        images, attribute_dict, subject_dates_dict = load_data(data_dir, num_of_slices=num_of_slices,
                                                               split=split,
                                                               keep_only_screening=keep_only_screening)

        self.attributes, indices_to_remove = load_extra_attributes(csv_path, attributes=attribute_size.keys(),
                                                                   attribute_dict=attribute_dict, subject_dates_dict=subject_dates_dict,
                                                                   keep_only_screening=keep_only_screening)
        self.attributes['slice'] = np.delete(self.attributes['slice'], indices_to_remove, axis=0)

        self.images = self.pad(torch.as_tensor(np.delete(images, indices_to_remove, axis=0).copy(), dtype=torch.float32).unsqueeze(1))

        if normalize_:
            self.attributes = {attr: normalize(torch.tensor(np.array(values), dtype=torch.float32), attr) for attr, values in self.attributes.items()}
        else:
            self.attributes = {attr: torch.tensor(np.array(values), dtype=torch.float32) for attr, values in self.attributes.items()}

        self.attrs = torch.cat([self.attributes[attr].unsqueeze(1) if len(self.attributes[attr].shape) == 1 else self.attributes[attr]
                                for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()}

        bins = np.linspace(0, 1, 5)
        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ["sex", "apoE", "slice"]:
                data = values.numpy()
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx], self.attrs[idx])

        if self.transform_cls:
            return self.transform_cls(self.images[idx]), self.attrs[idx]

        return self.images[idx], self.attrs[idx]
