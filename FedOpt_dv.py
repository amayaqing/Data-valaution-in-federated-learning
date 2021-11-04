from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
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
from RDV import RDV

tff.backends.reference.set_reference_context()

update_method = 'posi'  # posi, posi_weighted, normal


def model_update_method(method, model, local_models, local_weights, test_images, test_labels_onehot):
    # indicate model aggregation method:
    # 'normal' denotes normal average aggregation; 'posi' denotes 'positive-only', 'posi_weighted' denotes 'positive-weighted' strategy
    if method == 'normal':
        model_g = model_aggregate(model, local_models, local_weights)
    else:
        gradient_weights_local, gradient_biases_local = calc_gradiant(model, local_models)
        agent_sv = RDV(model, local_models, local_weights, test_images, test_labels_onehot)
        if method == 'posi':
            posi_p = [i for i in range(len(agent_sv)) if agent_sv[i] > 0]  # participants with positive shapley value
            print("positive contributors: ", posi_p)
            model_g = model_aggregate(model, [local_models[i] for i in posi_p], [local_weights[i] for i in posi_p])
        elif method == 'posi_weighted':
            posi_p = [i for i in range(len(agent_sv)) if agent_sv[i] > 0]  # participants with positive shapley value
            print("positive contributors: ", posi_p)
            agent_sv_posi = [sv for sv in agent_sv if sv > 0]
            sv_sum = reduce(lambda x, y: x + y, agent_sv_posi)
            sv_weights = [n / sv_sum for n in agent_sv_posi]
            print("sv_weights: ", sv_weights)
            model_g = model_aggregate_weighted(model, [local_models[i] for i in posi_p], sv_weights)
        else:
            model_g = init_model()
            print("null method!")

    return model_g



if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    start_time = time.time()

    cumu_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    round_group_sv = np.zeros([NUM_AGENT, NUM_ROUND], dtype=np.float32)
    accuracys = []
    round_time = []

    # load data and distribute
    mnist_train, test_images, test_labels_onehot = load_data()
    local_num_data = [NUM_LOCAL_DATA] * NUM_AGENT
    federated_train_data = distribute_data(mnist_train, local_num_data)
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]

    # for noise environment and unbalanced environment creation
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
        print("learning rate: ", learning_rate)

        # update model
        model = model_update_method(update_method, model, local_models, local_weights, test_images, test_labels_onehot)

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
