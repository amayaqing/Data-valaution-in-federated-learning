from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import collections
import matplotlib.pyplot as plt
from functools import reduce

import nest_asyncio
nest_asyncio.apply()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('DV')
sys.path.append('Util')

from glob import *
from util import *
from plot import *
from RDV import RDV
from ClusterDV import clusterDV
from TmcDV import tmcDV
from KsubDV import ksubDV
from DefDV import defDV

tff.backends.reference.set_reference_context()


if __name__ == "__main__":

    cumu_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    round_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    accuracys = []
    round_time = []

    # load data
    mnist_train, test_images, test_labels_onehot = load_data()
    local_num_data = [NUM_LOCAL_DATA] * NUM_AGENT
    federated_train_data = distribute_data(mnist_train, local_num_data)
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]

    # create noisy or unbalanced dataset
    if NOISE_ADD == True:
        noise = [0, 0, 0.35, 0.65, 1]
        # noise = [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 1.2, 1.2]
        federated_train_data, local_num_data, num_data, local_weights = prepare_data_noise(mnist_train, federated_train_data, noise)

    if UNBALANCE == True:
        local_num_data = [100, 100, 1000, 1000, 10000]
        #local_num_data = [100, 100, 500, 500, 1000, 1000, 2000, 2000, 5000, 5000]
        federated_train_data, local_num_data, num_data, local_weights = prepare_data_unbalanced(mnist_train, federated_train_data, local_num_data)
    print("local_num_data:", local_num_data)
    print("local_weights:", local_weights)


    model = init_model()

    for round_num in range(NUM_ROUND):

        start_time = time.time()

        # add noise in the middle
        if NOISE_ADD_LATER == True and round_num == 6:
            noise = [0, 0, 0.35, 0.65, 1]
            federated_train_data, local_num_data, num_data, local_weights = prepare_data_noise(mnist_train, federated_train_data, noise)

        # P5 add more data in the middle
        if UNBALANCED_LATER == True and round_num == 6:
            local_num_data = local_num_data = [1000, 1000, 1000, 1000, 30000]
            federated_train_data, local_num_data, num_data, local_weights = prepare_data_unbalanced(mnist_train, federated_train_data, local_num_data)
            print("After add number: ")
            print("local_num_data:", local_num_data)
            print("local_weights:", local_weights)

        # local training
        local_models = federated_train(model, learning_rate, federated_train_data)     # local_models return models for everyone, model[user]['weights'/'bias']
        print("learning rate: ", learning_rate)
        # get local model gradient
        gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)

        if dv_method == 'RDV':
            agent_sv = RDV(model, local_models, local_weights, test_images, test_labels_onehot)
            path = 'Image/RDV/'
        elif dv_method == 'cluster':
            agent_sv = clusterDV(model, local_models, local_weights, test_images, test_labels_onehot, cluster_num)
            path = 'Image/ClusterDV/'
        elif dv_method == 'tmc':
            agent_sv = tmcDV(model, local_models, local_weights, test_images, test_labels_onehot)
            path = 'Image/TmcDV/'
        elif dv_method == 'ksub':
            agent_sv = ksubDV(model, local_models, local_weights, test_images, test_labels_onehot)
            path = 'Image/KsubDV/'
        elif dv_method == 'def':
            agent_sv = defDV(model, test_images, test_labels_onehot, mnist_train, local_num_data)
            path = 'Image/DefDV/'

        # calcualte Shapley value
        round_group_sv, cumu_group_sv = record_sv(agent_sv, round_group_sv, cumu_group_sv, round_num)

        # update model
        model = model_aggregate(model, local_models, local_weights)
        # accuracy measure
        acc = model_accuracy(model, test_images, test_labels_onehot)
        accuracys.append(acc)
        # time measure
        rt = time.time() - start_time
        round_time.append(rt)
        print('round {}, accuracy={}, time={}'.format(round_num, acc, rt))

        # change learning rate for each round
        learning_rate = learning_rate * 0.9


    for i in range(NUM_AGENT):
        print("\nCumulative SV for Client {}: {}".format(i, cumu_group_sv[i]))
    print('\naccuracys:', accuracys)
    print("\ntime:", round_time)
    print("\naverage time:", np.mean(round_time))

    plot_util(path, round_group_sv, cumu_group_sv)