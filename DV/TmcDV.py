from __future__ import absolute_import, division, print_function
import numpy as np
import random

from glob import *
from util import *


def tmcDV(model, local_models, local_weights, test_images, test_labels_onehot):
    gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)

    agent_sv = np.zeros(NUM_AGENT)
    all_sets = []
    all_sets_acc = []
    performance_tolerance = 0.0001       # the threshold to stop calculation
    k = 80                               # repeat times
    p = [i for i in range(NUM_AGENT)]
    start_acc = train_with_gradient_and_valuation([], model, gradient_weights_local, gradient_biases_local,
                                                  local_weights, test_images, test_labels_onehot)
    final_acc = train_with_gradient_and_valuation(p, model, gradient_weights_local, gradient_biases_local,
                                                  local_weights, test_images, test_labels_onehot)

    for k_num in range(1, k + 1):
        random.shuffle(p)
        prev_acc = start_acc

        for i in range(1, len(p) + 1):
            curr = p[:i]

            if abs(final_acc - prev_acc) > performance_tolerance:
                if curr in all_sets:
                    v = all_sets_acc[all_sets.index(curr)]
                else:
                    v = train_with_gradient_and_valuation(curr, model, gradient_weights_local, gradient_biases_local,
                                                          local_weights, test_images, test_labels_onehot)
                    all_sets.append(curr)
                    all_sets_acc.append(v)

                # non-stratified calculation
                agent_sv[curr[-1]] = ((k_num - 1) / k_num) * agent_sv[curr[-1]] + (1 / k_num * (v - prev_acc))
                prev_acc = v

            else:
                # after achieving the performance threshold, assign 0 as the value to the rest data samples for this permutation
                agent_sv[curr[-1]] = ((k_num - 1) / k_num) * agent_sv[curr[-1]]

    return agent_sv