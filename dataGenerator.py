# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy

# <codecell>

def makeTransitionMatrix(numState, minSigma):
    # Input:
    # numState : number of hidden state
    # minSigma : the smallest singular value
    
    # Output: a transation matrix T satifying input specifications
    #       T is column stochastic. T_ij is state j to state i. 
    min_s = 0
    while min_s < minSigma:
        T_raw = np.random.rand(numState,numState) + np.identity(numState)
        T = np.array([row *1.0 / sum(row) for row in T_raw])
        T = np.transpose(T)
        U, s, V = np.linalg.svd(T, full_matrices=True)
        min_s = min(s)
    return T

# <codecell>

makeTransitionMatrix(5, 0.1)

# <codecell>

def makeDistribution(numState):
    # generate a distribution over hidden states
    d = np.random.rand(numState)
    return d*1.0/sum(d)

# <codecell>

def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

# <codecell>

def generateData(N, numState, T, p, initDist, l):
    ### This function generates l methylation counts triples, 0 <= counts <= N.
    ### Data generated from an HMM with
    #   T is column stochastic
    
    # create l triples (x1, x2, x3), x1, x2, x3 \in [N].
    # T = np.transpose(makeTransitionMatrix(numState, min_sigma))
    T = np.transpose(T)
    values = np.arange(N+1)
    # p[] : methylation probability for hidden state
    # p = np.random.rand(numState)
    # c : list of possible hidden states
    c = np.arange(numState)
    
    
    ### Generate hidden state list
    # generate initial state
    initS = weighted_values(c, initDist, 1)
    h_lst = initS
    for i in range(l+2):
        h_curr = h_lst[-1]
        dist = T[h_curr]
        h_next = weighted_values(c, dist, 1)
        h_lst = np.append(h_lst, h_next)
    
    ### Generate observations
    x_lst = np.array([], dtype='int32')
    for i in range(len(h_lst)):
        q = p[h_lst[i]]
        x = np.random.binomial(N, q)
        x_lst = np.append(x_lst, x)
    
    ### Create triple list   
    data = np.array(zip(x_lst[0:l], x_lst[1:l+1], x_lst[2:l+2]))
    return data
    

# <codecell>

N = 10000
numState = 10
l = 10000
min_sigma = 0.1
T = makeTransitionMatrix(numState, min_sigma)
initDist = makeDistribution(numState)
p = np.random.rand(numState)
Data = generateData(N, numState, T, p, initDist, l)
Data

# <codecell>


# <codecell>


