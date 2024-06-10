import torch

def get_attribute_ids(attribute_size):
    attribute_indices = {}
    idx = 0
    for attr, size in attribute_size.items():
        attribute_indices[attr] = list(range(idx, idx + size))
        idx += size
    return attribute_indices

class SelectParentAttributesTransform:
    def __init__(self, name, attribute_size, graph_structure):
        self.name = name
        attribute_indices = get_attribute_ids(attribute_size)

        self.attr_ids = attribute_indices[name] if self.name != 'image' else None
        self.pa_ids = sum([attribute_indices[attr] for attr in graph_structure[name]], [])

    def __call__(self, img, attrs):
        if self.name == 'image':
            return img, torch.Tensor([attrs[idx] for idx in self.pa_ids])
        else:
            return torch.Tensor([attrs[idx] for idx in self.attr_ids]), torch.Tensor([attrs[idx] for idx in self.pa_ids])

class ReturnDictTransform:
    def __init__(self, attribute_size):
        self.attribute_ids = get_attribute_ids(attribute_size)

    def __call__(self, img, attrs):
        return {**{"image": img}, **{attr: attrs[ids] for attr, ids in self.attribute_ids.items()}}
