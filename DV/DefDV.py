from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from functools import reduce

from glob import *
from util import *


def defDV(model, test_images, test_labels_onehot, mnist_train, local_num_data):
    all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
    group_sv = []
    for s in all_sets:
        federated_train_data = []
        for item in s:
            federated_train_data_divide = [get_data_for_federated_agents(mnist_train, d, local_num_data) for d in
                                           range(NUM_AGENT)]
            federated_train_data.append(federated_train_data_divide[item])
        # retrain the model and get the accuracy
        current_model = federated_train_and_aggregate(model, learning_rate, federated_train_data)
        current_acc = model_accuracy(current_model, test_images, test_labels_onehot)
        group_sv.append(current_acc)

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
