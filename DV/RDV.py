from __future__ import absolute_import, division, print_function
from scipy.special import comb, perm

from glob import *
from util import *


def RDV(model, local_models, local_weights, test_images, test_labels_onehot):
    # calculate gradient
    gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)

    # list all combinations
    all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
    group_sv = []
    for s in all_sets:
        group_sv.append(
            # gradient descent to aggregate local models to global one and get the accuracy value
            train_with_gradient_and_valuation(s, model, gradient_weights_local, gradient_biases_local, local_weights,
                                              test_images, test_labels_onehot)
        )

    # calculate Shapley value
    agent_sv = []
    for index in range(NUM_AGENT):
        shapley = 0.0
        for j in all_sets:
            if index in j:
                remove_list_index = remove_list_indexed(index, j, all_sets)
                if remove_list_index != -1:
                    shapley += (group_sv[shapley_list_indexed(j, all_sets)] - group_sv[
                        remove_list_index]) / (comb(NUM_AGENT - 1, len(all_sets[remove_list_index])))
        agent_sv.append(shapley)

    return agent_sv