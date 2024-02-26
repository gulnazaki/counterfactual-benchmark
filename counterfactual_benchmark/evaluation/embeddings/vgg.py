from torchvision.models import vgg16, VGG16_Weights
import torch

def vgg(pretrained='False'):

    if not pretrained:
        torch.manual_seed(42)

    weights = VGG16_Weights.DEFAULT if pretrained else None
    model=vgg16(weights=weights)
    if torch.cuda.is_available():
        model.to("cuda")
    model.classifier = model.classifier[:-1]

    return model
