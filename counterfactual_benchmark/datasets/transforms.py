import torch

class SelectAttributesTransform:
    def __init__(self, attr_idx, pa_idx):
        self.attr_idx = attr_idx
        self.pa_idx = pa_idx

    def __call__(self, img, attrs):
        return attrs[[self.attr_idx]], torch.Tensor([attrs[idx] for idx in self.pa_idx])