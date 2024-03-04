import numpy as np
from tqdm import tqdm

def l1_distance(embedding_1, embedding_2, array=False):
    return np.mean(np.abs(embedding_1 - embedding_2), axis = (1 if array else 0))

def prob(div, S, S_leq=True):
    num  = (S <= div if S_leq == True else S >= div).sum()
    return num / len(S)

def same(p1, p2, name, bins):
    if name in ['thickness', 'intensity']:
        return np.searchsorted(bins[name], p1) == np.searchsorted(bins[name], p2)
    elif name == "digit":
        return np.argmax(p1) == np.argmax(p2)
    else:
        return p1[0] == p2[0]

def minimality(real, generated, interventions, bins):
    minimality_scores = []

    for factual, counterfactual, i in tqdm(zip(real, generated, interventions)):
        f, f_pa = factual
        cf, cf_pa = counterfactual

        div = l1_distance(f, cf)

        fs, cfs = [], []
        for r, r_pa in real:
            if same(r_pa, f_pa, i, bins):
                fs.append(r)
            elif same(r_pa, cf_pa, i, bins):
                cfs.append(r)

        if len(cfs) == 0 or len(fs) == 0:
            print(f"Warning: fs or cfs is empty for intervention {i}, f_pa {f_pa}, cf_pa {cf_pa}")
            continue

        S_f = l1_distance(np.array(fs), f, array=True)
        S_cf = l1_distance(np.array(cfs), f, array=True)

        minimality_scores.append(np.log(np.exp(prob(div, S_cf, S_leq=True)) + np.exp(prob(div, S_f, S_leq=False))))

    return minimality_scores
