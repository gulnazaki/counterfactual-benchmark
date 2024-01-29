def composition(factual_batch, method, parents, cycles=10):
    for i in range(cycles):
        abducted_noise = method.encode(factual_batch, parents)
        counterfactual_batch = method.decode(abducted_noise, parents)
        factual_batch = counterfactual_batch
    # loop on different embeddings
    l1_distance(factual_batch, counterfactual_batch) # add more distances

def l1_distance(factual_batch, counterfactual_batch):
    pass