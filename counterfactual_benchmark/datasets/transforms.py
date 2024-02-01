import torch

class SelectAttributesTransform:
    def __init__(self, name, attribute_size, graph_structure):
        attribute_indices = {}
        idx = 0
        for attr, size in attribute_size.items():
            attribute_indices[attr] = list(range(idx, idx + size))
            idx += size

        self.attr_ids = attribute_indices[name]
        self.pa_ids = sum([attribute_indices[attr] for attr in graph_structure[name]], [])

    def __call__(self, img, attrs):
        return torch.Tensor([attrs[idx] for idx in self.attr_ids]), torch.Tensor([attrs[idx] for idx in self.pa_ids])

class ReturnDictTransform:
    def __init__(self, attribute_size):
        self.attribute_ids = {}
        idx = 0
        for attr, size in attribute_size.items():
            self.attribute_ids[attr] = list(range(idx, idx + size))
            idx += size

    def __call__(self, img, attrs):
        return {**{"image": img}, **{attr: attrs[ids] for attr, ids in self.attribute_ids.items()}}
