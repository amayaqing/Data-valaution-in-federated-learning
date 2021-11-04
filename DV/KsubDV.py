from __future__ import absolute_import, division, print_function
import numpy as np
import time
from scipy.special import comb, perm
import random
import math
import copy

from glob import *
from util import *


def loc_all_sets(ll, l):        # find the test results in the calculation record
    for i in range(len(ll)):
        if set(ll[i]) == set(l):
            return i
    return -1

def ksubDV(model, local_models, local_weights, test_images, test_labels_onehot):
    gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)

    agent_sv = np.zeros(NUM_AGENT)
    s = 0.6  # fraction to sample in each position
    p = [i for i in range(NUM_AGENT)]
    all_sets = []
    all_sets_acc = []

    for index in range(NUM_AGENT):
        tmp_p = copy.deepcopy(p)
        tmp_p.remove(index)
        for posi in range(NUM_AGENT):
            sample_size = math.ceil(s * comb(NUM_AGENT - 1, posi))
            for _ in range(sample_size):
                # random sample
                random.seed(time.time())
                prev_l = random.sample(tmp_p, posi)

                # record the train-and-test result to avoid re-calculation
                loc = loc_all_sets(all_sets, prev_l)
                if loc != -1:
                    prev_l_acc = all_sets_acc[loc]
                else:
                    prev_l_acc = train_with_gradient_and_valuation(prev_l, model, gradient_weights_local,
                                                                   gradient_biases_local, local_weights,
                                                                   test_images, test_labels_onehot)
                    all_sets.append(prev_l)
                    all_sets_acc.append(prev_l_acc)

                l = copy.deepcopy(prev_l)
                l.append(index)
                loc = loc_all_sets(all_sets, l)
                if loc != -1:
                    l_acc = all_sets_acc[loc]
                else:
                    l_acc = train_with_gradient_and_valuation(l, model, gradient_weights_local,
                                                              gradient_biases_local, local_weights, test_images,
                                                              test_labels_onehot)
                    all_sets.append(l)
                    all_sets_acc.append(l_acc)

                # stratified calculation
                agent_sv[index] += 1 / sample_size * (l_acc - prev_l_acc)

    return agent_sv