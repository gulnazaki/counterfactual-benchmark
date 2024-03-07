import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from tqdm import tqdm
import sys
sys.path.append("../../")
from models.utils import rgbify

def fid(real_images, generated_images):
    metric = FID(normalize=True, reset_real_features=False).set_dtype(torch.float64).to('cuda')
    for real_batch in tqdm(real_images):
        metric.update(rgbify(real_batch["image"]).to('cuda'), real=True)

    for generated_batch in tqdm(generated_images):
        metric.update(rgbify(generated_batch).to('cuda'), real=False)

    fid_score = metric.compute().cpu().numpy()

    return fid_score
