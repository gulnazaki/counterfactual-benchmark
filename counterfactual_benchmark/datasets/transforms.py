import torch

class SelectAttributesTransform:
    def __init__(self, attr_idx, pa_idx):
        self.attr_idx = attr_idx
        self.pa_idx = pa_idx

    def __call__(self, img, attrs):
        return attrs[[self.attr_idx]], torch.Tensor([attrs[idx] for idx in self.pa_idx])

class ReturnLabelsTransform:
    def __init__(self, attributes, image_name='image'):
        self.attributes = attributes
        self.image_name = image_name

    def __call__(self, img, attrs):
        label_dict = {attr: value.unsqueeze(dim=-1) for attr, value in zip(self.attributes, attrs)}
        label_dict[self.image_name] = img
        return label_dict