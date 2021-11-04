from __future__ import absolute_import, division, print_function
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.special import comb, perm
from functools import reduce
from sklearn.cluster import KMeans

from glob import *
from util import *

def clusterDV(model, local_models, local_weights, test_images, test_labels_onehot, k):
    # predict
    local_preds = []
    for lm in local_models:
        m = np.dot(test_images, np.asarray(lm['weights']))
        test_result = m + np.asarray(lm['bias'])
        y = tf.nn.softmax(test_result)
        pred = tf.argmax(y, 1).numpy()
        local_preds.append(pred)

    # K-means partipiton based on p
    clusters = []
    local_preds = np.asarray(local_preds)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(local_preds)
    for c in range(k):
        clusters.append([i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == c])
    print("clusters:", clusters)

    # calculate cluster models
    cluster_models = []
    cluster_weights = []
    for c in range(k):
        model_c = model_aggregate(model, [local_models[i] for i in clusters[c]],
                                  [local_weights[i] for i in clusters[c]])
        weights_sum = reduce(lambda x, y: x + y, [local_weights[i] for i in clusters[c]])
        cluster_models.append(model_c)
        cluster_weights.append(weights_sum)

    cluster_gradient_weights, cluster_gradient_biases = calc_gradiant(model, cluster_models)

    # Shapley
    all_sets = PowerSetsBinary([i for i in range(k)])
    group_sv = []
    for s in all_sets:
        group_sv.append(
            train_with_gradient_and_valuation(s, model, cluster_gradient_weights, cluster_gradient_biases,
                                              cluster_weights, test_images, test_labels_onehot)
        )

    cluster_sv = []
    for index in range(k):
        shapley = 0.0
        for j in all_sets:
            if index in j:
                remove_list_index = remove_list_indexed(index, j, all_sets)
                if remove_list_index != -1:
                    shapley += (group_sv[shapley_list_indexed(j, all_sets)] - group_sv[
                        remove_list_index]) / (comb(NUM_AGENT - 1, len(all_sets[remove_list_index])))
        cluster_sv.append(shapley)

    # cluster sv --> individual sv
    agent_sv = np.zeros([NUM_AGENT], dtype=np.float32)
    for c in range(k):
        for i in range(len(clusters[c])):
            agent_sv[clusters[c][i]] = cluster_sv[c] / len(clusters[c])

    return agent_sv