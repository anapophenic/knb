# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy
import binom_hmm as bh

def generate_p(m):
    #p = np.asarray([0,0.5,1])
    #p = np.asarray([0.3,0.7])
    p = np.asarray(bh.unif_partition(m));
    #print p
    return p


def generate_p_N(N):
    p = np.ones(N+1) / (N+1);
    return p

def generate_T(m, min_sigma_t):
    #T = dataGenerator.makeTransitionMatrix(m, min_sigma_t)
    #T = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]);
    #T = np.eye(3);
    #T = np.asarray([[0.8, 0.2], [0.2, 0.8]])

    T = min_sigma_t * np.eye(m) + (1 - min_sigma_t) * np.random.random((m, m))

    return bh.col_normalize(T)

def generate_pi(m):
    #pi = dataGenerator.makeDistribution(m)
    #pi = np.asarray([0.33,0.33,0.34])
    #pi = np.asarray([0.6, 0.4])
    pi = np.ones(m) / m;

    return pi

def generate_O(m, n, min_sigma_o):

    O = min_sigma_o * np.eye(n,m) + (1 - min_sigma_o) * np.random.random((n, m))

    return bh.col_normalize(O)

def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

def generate_seq(T, O, pi, l):

    m = np.shape(pi)[0];
    n = np.shape(O)[0];

    h = np.int_(np.zeros(l));
    x = np.int_(np.zeros(l));

    for i in range(l):
        if i == 0:
            h[i] = weighted_values(range(m), pi, 1);
        else:
            h[i] = weighted_values(range(m), T[:,h[i-1]].reshape(m), 1);

    for i in range(l):
        x[i] = weighted_values(range(n), O[:,h[i]].reshape(n), 1);

    return x, h

def generate_firstfew(T, O, pi, l):
    ### This function generates l triples, each representing methylation counts at first 3 sites of a sequence
    ### Data generated from an HMM with transition matrix T and obervation matrix O
    #   T and O are column stochastic
    x_zipped = [];

    for i in range(l):
        x, h = generate_seq(T, O, pi, 3)
        x_zipped.append((x[0], x[1], x[2]))

    return x_zipped

def generate_longchain(T, O, pi, l):

    x, h = generate_seq(T, O, pi, l+2);
    x_zipped = np.array(zip(x[0:l], x[1:l+1], x[2:l+2]))

    return x_zipped
