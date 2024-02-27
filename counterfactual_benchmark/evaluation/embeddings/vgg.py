from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
import torch

def vgg_normalize(input):
    # [0, 255] -> [0, 1]
    in_0_1 = input / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    return normalize(in_0_1)


def vgg(pretrained='False'):

    if not pretrained:
        torch.manual_seed(42)

    weights = VGG16_Weights.DEFAULT if pretrained else None
    model=vgg16(weights=weights)
    if torch.cuda.is_available():
        model.to("cuda")
    model.classifier = model.classifier[:-1]

    return model
