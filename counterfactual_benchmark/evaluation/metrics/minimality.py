import numpy as np
from tqdm import tqdm

def l1_distance(embedding_1, embedding_2, array=False):
    return np.mean(np.abs(embedding_1 - embedding_2), axis = 1 if array else 0)

def prob(div, S, S_leq=True):
    total = len(S)
    if total == 0:
        print("Warning: S is empty")
        return 0
    else:
        num  = (S <= div if S_leq == True else S >= div).sum()
        return num / len(S)

def same(p1, p2, name, bins):
    if name in ['thickness', 'intensity']:
        return np.searchsorted(bins[name], p1[name]) == np.searchsorted(bins[name], p2[name])
    elif name == "digit":
        return np.argmax(p1[name]) == np.argmax(p2[name])
    else:
        return p1[name][0] == p2[name][0]

def minimality(feat_dict, bins):

    minimality_scores = []

    for factual, counterfactual, i in tqdm(zip(zip(*feat_dict['real']), zip(*feat_dict['counterfactual']), feat_dict['interventions']), total=len(feat_dict['interventions'])):
        f, f_pa = factual
        cf, cf_pa = counterfactual

        div = l1_distance(f, cf)

        fs, cfs = [], []
        for real, real_pa in zip(*feat_dict['real']):
            if same(real_pa, f_pa, i, bins):
                fs.append(real)
            elif same(real_pa, cf_pa, i, bins):
                cfs.append(real)

        S_f = l1_distance(np.array(fs), f, array=True)
        S_cf = l1_distance(np.array(cfs), f, array=True)

        minimality_scores.append(np.log(np.exp(prob(div, S_cf, S_leq=True)) + np.exp(prob(div, S_f, S_leq=False))))

    print(f"Minimality score: mean {round(np.mean(minimality_scores), 3)}, std {round(np.std(minimality_scores), 3)}")
