from vgg import vgg, vgg_normalize
from classifier_embeddings import ClassifierEmbeddings
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import sys
sys.path.append("../../")
from models.utils import rgbify


def get_embedding_model(embedding, pretrained_vgg, classifier_config=None):
    if embedding == "vgg":
        embedding_model = vgg(pretrained_vgg)
    elif embedding == "clfs":
        embedding_model = ClassifierEmbeddings(classifier_config)
    elif embedding == "lpips":
        embedding_model = LPIPS(net_type='vgg', normalize=True).to('cuda')
    else:
        embedding_model = None
    return embedding_model

def get_embedding_fn(embedding, unnormalize_fn, embedding_model):
    if embedding is None:
        embedding_fn = lambda x, _: unnormalize_fn(x, "image").cpu().numpy()
    elif embedding == "vgg":
        embedding_fn = lambda x, _: embedding_model(vgg_normalize(rgbify(x, normalized=True), to_0_1=False)).detach().cpu().numpy()
    elif embedding == "clfs":
        embedding_fn = lambda x, cond: embedding_model(x, cond, only_intensity=True).detach().cpu().numpy()
    elif embedding == "lpips":
        embedding_fn = lambda x, y: embedding_model(rgbify(x, normalized=True), rgbify(y, normalized=True)).detach().cpu().numpy()
    else:
        exit(f"Invalid embedding: {embedding}")

    return embedding_fn