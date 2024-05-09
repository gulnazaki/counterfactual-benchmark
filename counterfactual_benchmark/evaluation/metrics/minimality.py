import numpy as np
from tqdm import tqdm

W1, W2 = 0.8, 0.2

def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (np.square(np.exp(q_logscale)) + np.square(q_loc - p_loc))
        / np.square(np.exp(p_logscale))
    )

def kl_divergence(embedding_1, embedding_2):
    q_loc, q_logscale = (embedding_1[:, 0], embedding_1[:, 0]) if len(embedding_1.shape) == 3 else (embedding_1[0], embedding_1[0])
    p_loc, p_logscale = (embedding_2[:, 0], embedding_2[:, 0]) if len(embedding_2.shape) == 3 else (embedding_2[0], embedding_2[0])
    kl_pp = gaussian_kl(q_loc, q_logscale, p_loc, p_logscale)
    return np.sum(kl_pp, axis=-1) / embedding_1.shape[-1]

def l1_distance(embedding_1, embedding_2):
    return np.mean(np.abs(embedding_1 - embedding_2), axis=1)

def prob(div, S, S_leq=True):
    num  = np.sum(S <= div) if S_leq == True else np.sum(S >= div)
    return num / len(S)

def same(p1, p2, name, bins):
    if name in ['thickness', 'intensity']:
        return np.searchsorted(bins[name], p1) == np.searchsorted(bins[name], p2)
    else:
        return p1 == p2

def minimality(real, generated, interventions, bins, embedding):
    real_features = np.array([r[0] for r in real])
    generated_features = np.array([g[0] for g in generated])
    real_parents = np.array([r[1][0] if r[1].shape[0] == 1 else np.argmax(r[1]) for r in real])
    generated_parents = np.array([g[1][0] if g[1].shape[0] == 1 else np.argmax(g[1]) for g in generated])

    minimality_scores = []
    prob1s = []
    prob2s = []

    distance_fn = kl_divergence if embedding == 'vae' else l1_distance

    divs = distance_fn(generated_features, real_features)

    for f, f_pa, cf_pa, i, div in tqdm(zip(real_features, real_parents, generated_parents, interventions, divs), total=len(interventions)):

        fs_mask = np.array([same(r_pa, f_pa, i, bins) for r_pa in real_parents])
        cfs_mask = np.array([same(r_pa, cf_pa, i, bins) for r_pa in real_parents])

        fs = real_features[fs_mask]
        cfs = real_features[cfs_mask]

        if len(cfs) == 0 or len(fs) == 0:
            print(f"Warning: fs or cfs is empty for intervention {i}, f_pa {f_pa}, cf_pa {cf_pa}")
            continue

        S_f = distance_fn(fs, f)
        S_cf = distance_fn(cfs, f)

        prob1 = prob(div, S_cf, S_leq=True)
        prob2 = prob(div, S_f, S_leq=False)

        prob1s.append(prob1)
        prob2s.append(prob2)
        minimality_scores.append(np.log(W1 * np.exp(prob1) + W2 * np.exp(prob2)))

    return minimality_scores, prob1s, prob2s
