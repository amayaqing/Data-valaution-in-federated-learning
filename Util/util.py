from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import collections
import matplotlib.pyplot as plt
from functools import reduce
import os
from glob import *

BATCH_TYPE = tff.StructType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.StructType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))])
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)
SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


def get_data_for_federated_agents(source, num, local_num_data):
    # allocate data to local participants
    output_sequence = []
    source_size = source[0].shape[0]

    if NON_IID == True:
        Samples = []
        for digit in range(0, 10):
            samples = [i for i, d in enumerate(source[1]) if d == digit]
            Samples.append(samples)

        digit_start = int(num * 2 % 10)
        le = int(int(num / 5) * NUM_LOCAL_DATA / 2)
        ri = int(le + NUM_LOCAL_DATA / 2)

        all_samples = []
        for sample in Samples[digit_start : digit_start+2]:
            if le%source_size < ri%source_size:
                for i in range(le%source_size, ri%source_size):
                    all_samples.append(sample[i])
            else:
                for i in range(le%source_size, source_size):
                    all_samples.append(sample[i])
                for i in range(0, ri%source_size):
                    all_samples.append(sample[i])

    else:   # IID
        # allocate data in order to avoid overlapping (note: inspired by the demo question)
        le = int(np.sum(local_num_data[:num]))
        ri = int(le + local_num_data[num])

        if le%source_size < ri%source_size:
            all_samples = np.arange(le % source_size, ri % source_size)
        else:
            all_samples = np.arange(le % source_size, source_size)
            all_samples = np.append(all_samples, np.arange(0, ri % source_size))

    # batch
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    return output_sequence


def add_noise(source, client_data, noise_level):
    # add noise to local user data
    output_sequence = client_data
    noise_num = int(len(source[0])/NUM_AGENT * noise_level)
    all_samples = np.random.randint(len(source[0]), size=(noise_num))
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([(source[1][i]+np.random.randint(10))%10 for i in batch_samples], dtype=np.int32)})

    return output_sequence

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(batch.y, 10) * tf.log(predicted_y), axis=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    # Perform one step of gradient descent using loss from `batch_loss`.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    l = tff.sequence_reduce(all_batches, initial_model, batch_fn)
    return l


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    l = tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])
    return l

@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train_and_aggregate(model, learning_rate, data):
    return tff.federated_mean(
        tff.federated_map(local_train,
                          [tff.federated_broadcast(model),
                           tff.federated_broadcast(learning_rate),
                           data])
    )

def readTestImagesFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split("\t")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)

def readTestLabelsFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split(" ")
        for i in p:
            if i!="":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)

def model_aggregate(model, local_models, local_weights):
    # aggregate local updates to a global model
    m_w = np.zeros([784, 10], dtype=np.float32)
    m_b = np.zeros([10], dtype=np.float32)
    for i in range(len(local_models)):
        m_w = np.add(np.multiply(local_models[i]['weights'], 1/len(local_weights)), m_w)
        m_b = np.add(np.multiply(local_models[i]['bias'], 1/len(local_weights)), m_b)
    model_g = {
        'weights': m_w,
        'bias': m_b
    }
    return model_g

def model_aggregate_weighted(model, local_models, local_weights):
    # weighted aggregate local updates to a global model
    m_w = np.zeros([784, 10], dtype=np.float32)
    m_b = np.zeros([10], dtype=np.float32)
    for i in range(len(local_models)):
        m_w = np.add(np.multiply(local_models[i]['weights'], local_weights[i]), m_w)
        m_b = np.add(np.multiply(local_models[i]['bias'], local_weights[i]), m_b)
    model_g = {
        'weights': m_w,
        'bias': m_b
    }
    return model_g

def model_accuracy(model_g, test_images, test_labels_onehot):
    # test the model and get the accuracy
    m = np.dot(test_images, np.asarray(model_g['weights']))
    test_result = m + np.asarray(model_g['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()

def train_with_gradient_and_valuation(agent_list, model, gradient_weights_local, gradient_biases_local, local_weights, test_images, test_labels_onehot):
    # aggregate
    model_g = {
        'weights': model['weights'],
        'bias': model['bias']
    }
    gradient_w = np.zeros([784, 10], dtype=np.float32)
    gradient_b = np.zeros([10], dtype=np.float32)
    for j in agent_list:
        gradient_w = np.add(np.multiply(gradient_weights_local[j], 1 / len(agent_list)),
                            gradient_w)  # weighted averagage
        gradient_b = np.add(np.multiply(gradient_biases_local[j], 1 / len(agent_list)), gradient_b)
    model_g['weights'] = np.subtract(model_g['weights'], np.multiply(learning_rate, gradient_w))  # update
    model_g['bias'] = np.subtract(model_g['bias'], np.multiply(learning_rate, gradient_b))

    # test
    acc = model_accuracy(model_g, test_images, test_labels_onehot)
    return acc


def remove_list_indexed(removed_ele, original_l, ll):
    # get the record location of one combination
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1


def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1


def PowerSetsBinary(items):
    # list all combinations
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all

def load_data():
    # get mnist dataset
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    test_images = readTestImagesFromFile(False)
    test_labels_onehot = readTestLabelsFromFile(False)

    return mnist_train, test_images, test_labels_onehot


def distribute_data(mnist_train, local_num_data):
    federated_train_data_divide = [get_data_for_federated_agents(mnist_train, d, local_num_data) for d in
                                   range(NUM_AGENT)]
    return federated_train_data_divide


def prepare_data_noise(mnist_train, federated_train_data, noise):
    local_num_data = [(NUM_LOCAL_DATA * (1 + i)) for i in noise]
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]
    # add_noise
    for i in range(len(noise)):
        federated_train_data[i] = add_noise(mnist_train, federated_train_data[i], noise[i])
    print("After add noise:")
    for i in range(len(federated_train_data)):
        print(len(federated_train_data[i]), len(federated_train_data[i][0]['x']))
    return federated_train_data, local_num_data, num_data, local_weights


def prepare_data_unbalanced(mnist_train, federated_train_data, local_num_data):
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]
    # reload the data
    federated_train_data = distribute_data(mnist_train, local_num_data)
    return federated_train_data, local_num_data, num_data, local_weights

def add_agent(mnist_train, federated_train_data, local_num_data, round_group_sv, cumu_group_sv):
    NUM_AGENT += 1
    local_num_data.append(1000)
    num_data = reduce(lambda x, y: x + y, local_num_data)
    local_weights = [n / num_data for n in local_num_data]

    tmp = np.zeros([1, NUM_ROUND], dtype=np.float32)
    cumu_group_sv = np.concatenate((cumu_group_sv, tmp), axis=0)
    round_group_sv = np.concatenate((round_group_sv, tmp), axis=0)
    federated_train_data.append(get_data_for_federated_agents(mnist_train, NUM_AGENT - 1, local_num_data))

    return federated_train_data, local_num_data, num_data, local_weights, round_group_sv, cumu_group_sv


def init_model():
    # model initialization
    initial_model = collections.OrderedDict(
        weights=np.zeros([784, 10], dtype=np.float32),
        bias=np.zeros([10], dtype=np.float32)
    )
    return initial_model

def calc_gradiant(model, local_models):
    # calculate gradient of models
    gradient_weights_local = []
    gradient_biases_local = []
    for i in range(len(local_models)):
        gradient_weight = np.divide(np.subtract(model['weights'], local_models[i]['weights']), learning_rate)
        gradient_bias = np.divide(np.subtract(model['bias'], local_models[i]['bias']), learning_rate)
        gradient_weights_local.append(gradient_weight)
        gradient_biases_local.append(gradient_bias)
    return gradient_weights_local, gradient_biases_local

def record_sv(agent_sv, round_group_sv, cumu_group_sv, round_num):
    # record shapley value for each agent
    for i, ag_s in enumerate(agent_sv):
        if round_num == 0:
            cumu_group_sv[i][round_num] = ag_s
        else:
            cumu_group_sv[i][round_num] = cumu_group_sv[i][round_num - 1] + ag_s
        round_group_sv[i][round_num] = ag_s
        print("{}: {}".format(i, ag_s))
    return round_group_sv, cumu_group_sv
