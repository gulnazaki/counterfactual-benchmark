import numpy as np
import torch
from metrics.compositition import composition

class Evaluator():

    def evaluate(self, dataloader, method, metrics, parents):
        test_set = dataloader.load("test_set", self.batch_size)
        train_set = dataloader.load("train_set", self.batch_size)


        # composition
        composition_scores = []
        for factual_batch in test_set:
            composition_scores.append(composition(factual_batch["image"], method, factual_batch["attrs"], self.cycles))
        composition_score = np.mean(composition_scores)

        # get "true" counterfactuals
        counterfactuals = []
        for factual_batch in test_set:
            for do_pa in parents:
                idx = torch.randperm(train_set[do_pa].shape[0])
                interventions = train_set[do_pa].clone()[idx][:self.batch_size]
                abducted_noise = method.encode(factual_batch, factual_batch["attrs"])
                counterfactual_batch, counterfactual_parents = method.decode(abducted_noise, interventions)
                counterfactuals.append((counterfactual_batch, counterfactual_parents))
        # effectiveness
        effectiveness(counterfactuals, predictors)

        # coverage & density
        coverage_density(train_set["image"], counterfactuals[0])

