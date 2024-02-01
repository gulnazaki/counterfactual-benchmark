from ..embeddings.vgg import vgg
from .prdc import compute_prdc
import torchvision.transforms as T
import torch
import numpy as np


#def coverage_density(real_images, generated_images, k = 5,embedding_fn=vgg, pretrained=True):
def coverage_density(factuals, counterfactuals, k = 5, embedding_fn=vgg, pretrained=True): 
    
    transform28 = T.Resize(size = (28,28))
    transform224 = T.Resize(size = (224,224))
    
    model = embedding_fn(pretrained)
   
    generated_features = []
    real_features = []
    
    for image in counterfactuals:
        rgb_batch = np.repeat(image, 3, axis=1)
        input = transform28(rgb_batch)
        input = transform224(input)
        if torch.cuda.is_available():
           input = input.to("cuda")
        feat = model(input)
        generated_features.append(feat.cpu().detach().numpy())
       
    for image in factuals:
        rgb_batch = np.repeat(image, 3, axis=1)
        input = transform28(rgb_batch)
        input = transform224(input)
        if torch.cuda.is_available():
           input = input.to("cuda")
        feat = model(input)
        real_features.append(feat.cpu().detach().numpy())  
       
       
    for i in range(len(generated_features)):
        generated_features[i] = generated_features[i].flatten()
    generated_features = np.array(generated_features)
    
    for i in range(len(real_features)):
        real_features[i] = real_features[i].flatten()
    real_features = np.array(real_features)
    
    metrics = compute_prdc(real_features, generated_features, k)
    
    print ('Coverage: ', metrics['coverage'])
    print ('Density: ', metrics['density'])
    print ('Precision: ', metrics['precision'])
    print ('Recall: ', metrics['recall'])
    
    

 