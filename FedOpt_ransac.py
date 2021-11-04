from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from functools import reduce
import random

import nest_asyncio
nest_asyncio.apply()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('DV')
sys.path.append('Util')

from glob import *
from util import *
from RDV import RDV

tff.backends.reference.set_reference_context()

NUM_RANSAC = 20    # repeat n times for RANSAC
n = 3             # select n participants each time


if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    start_time = time.time()

    cumu_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    round_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    accuracys = []
    round_time = []

    mnist_train, test_images, test_labels_onehot = load_data()
    local_num_data = [NUM_LOCAL_DATA] * NUM_AGENT
    federated_train_data = distribute_data(mnist_train, local_num_data)
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]

    if NOISE_ADD == True:
        noise = [0, 0, 0.35, 0.65, 1]
        # noise = [0, 0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 1.2, 1.2]
        federated_train_data, local_num_data, num_data, local_weights = prepare_data_noise(mnist_train,
                                                                                           federated_train_data, noise)
    if UNBALANCE == True:
        local_num_data = [100, 100, 1000, 1000, 10000]
        # local_num_data = [100, 100, 500, 500, 1000, 1000, 2000, 2000, 5000, 5000]
        federated_train_data, local_num_data, num_data, local_weights = prepare_data_unbalanced(mnist_train,
                                                                                                federated_train_data,
                                                                                                local_num_data)
    print("local_num_data:", local_num_data)
    print("local_weights:", local_weights)

    model = init_model()
    acc = model_accuracy(model, test_images, test_labels_onehot)
    print("starting accuracy: ", acc)

    # train
    for round_num in range(NUM_ROUND):
        local_models = federated_train(model, learning_rate,
                                       federated_train_data)  # local_models return models for everyone, model[user]['weights'/'bias']
        gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)
        print("learning rate: ", learning_rate)

        agent_value = np.zeros(NUM_AGENT)
        agent_ntimes = np.zeros(NUM_AGENT)
        for ransac_num in range(NUM_RANSAC):
            # random sample participant set
            curr_p = random.sample([i for i in range(NUM_AGENT)], n)
            curr_p_acc = train_with_gradient_and_valuation(curr_p, model, gradient_weights_local, gradient_biases_local, local_weights, test_images, test_labels_onehot)
            # calcualte average
            for i in curr_p:
                agent_ntimes[i] += 1
                agent_value[i] += curr_p_acc / n

        # update model
        for i in range(len(agent_value)):
            if agent_ntimes[i] != 0:
                agent_value[i] = agent_value[i] / agent_ntimes[i]  # calculate mean
        print(agent_value, agent_ntimes)
        # sort and select top-n participants
        agent_sort = np.argsort(agent_value)
        agg_p = [i for i in agent_sort[-n:] if agent_value[i] > 0]
        print(agg_p)
        if len(agg_p) > 0:
            model = model_aggregate(model, [local_models[i] for i in agg_p], [local_weights[i] for i in agg_p])


        # accuracy measure
        acc = model_accuracy(model, test_images, test_labels_onehot)
        accuracys.append(acc)
        # time measure
        rt = time.time() - start_time
        round_time.append(rt)

        print('round {}, accuracy={}, time={}'.format(round_num, acc, rt))

        learning_rate = learning_rate * 0.9

    acc_avg = np.array(accuracys) / np.array(round_time)

    print('\naccuracys:', accuracys)
    print("\ntime:", round_time)
    print("\naccuracy/time:", acc_avg)
