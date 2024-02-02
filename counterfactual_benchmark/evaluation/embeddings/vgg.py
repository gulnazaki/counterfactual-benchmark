from torchvision.models import vgg16
import torch

def vgg(pretrained='False'):

    model=vgg16(pretrained=pretrained)
    if torch.cuda.is_available():
        model.to("cuda")
    model.classifier = model.classifier[:-1]
    
    return model

    
    