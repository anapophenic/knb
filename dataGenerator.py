# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy
import moments_cons as mc

def generate_p(m):
    #p = np.asarray([0,0.5,1])
    #p = np.asarray([0.3,0.7])
    p = np.asarray(mc.unif_partition(m));
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

    return mc.col_normalize(T)

def generate_pi(m):
    #pi = dataGenerator.makeDistribution(m)
    #pi = np.asarray([0.33,0.33,0.34])
    #pi = np.asarray([0.6, 0.4])
    pi = np.ones(m) / m;

    return pi

def generate_O(m, n, min_sigma_o):

    O = min_sigma_o * np.eye(n,m) + (1 - min_sigma_o) * np.random.random((n, m))

    return mc.col_normalize(O)

# <codecell>

def makeTransitionMatrix(m, minSigma):
    # Input:
    # m : number of hidden state
    # minSigma : the smallest singular value

    # Output: a transation matrix T satifying input specifications
    #       T is column stochastic. T_ij is state j to state i.
    min_s = 0
    while min_s < minSigma:
        T_raw = np.random.rand(m,m) + np.identity(m)
        T = np.array([row *1.0 / sum(row) for row in T_raw])
        T = np.transpose(T)
        U, s, V = np.linalg.svd(T, full_matrices=True)
        min_s = min(s)
    return T

# <codecell>

def makeObservationMatrix(m, numObs, minSigma):
    # Input:
    # m : number of hidden state
    # numObs   : number of possible observations
    # minSigma : the smallest singular value

    # Output: an observation matrix T satifying input specifications
    #       T is column stochastic. T_ij is state j to state i.
    assert (numObs >= m),"More hidden states than observations"
    min_s = 0
    while min_s < minSigma:
        T_raw = np.random.rand(m,numObs)
        T = np.array([row *1.0 / sum(row) for row in T_raw])
        T = np.transpose(T)
        U, s, V = np.linalg.svd(T, full_matrices=True)
        min_s = min(s)
    return T

# <codecell>

makeObservationMatrix(3, 5, 0.1)
#makeTransitionMatrix(5, 0.1)

# <codecell>

def makeDistribution(m):
    # generate a distribution over hidden states
    d = np.random.rand(m)
    return d*1.0/sum(d)

# <codecell>

def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

# <codecell>

def generate_seq(T, O, pi, l):

    m = np.shape(pi)[0];
    n = np.shape(O)[0];

    h = [0] * l;
    x = [0] * l;

    for i in range(l):
        if i == 0:
            h[i] = weighted_values(range(m), pi, 1);
        else:
            h[i] = weighted_values(range(m), T[:,h[i-1]].reshape(m), 1);

    for i in range(l):
        x[i] = weighted_values(range(n), O[:,h[i]].reshape(n), 1);

    return x, h


def generateData_longChain(N, m, T, p, pi, l):
    ### This function generates l methylation counts triples, 0 <= counts <= N.
    ### Data generated from an HMM with transition matrix T
    #   T is column stochastic

    # create l triples (x1, x2, x3), x1, x2, x3 \in [N].
    # T = np.transpose(makeTransitionMatrix(m, min_sigma))
    T = np.transpose(T)
    values = np.arange(N+1)
    # p[] : methylation probability for hidden state
    # p = np.random.rand(m)
    # c : list of possible hidden states
    c = np.arange(m)


    ### Generate hidden state list
    # generate initial state
    initS = weighted_values(c, pi, 1)
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

def generateData_firstFew(N, m, T, p, pi, l):
    ### This function generates l triples, each representing methylation counts at first 3 cites of a sequence
    ### Data generated from an HMM with transition matrix T
    #   T is column stochastic

    # create l triples (x1, x2, x3), x1, x2, x3 \in [N].
    T = np.transpose(T)
    # values : list of possible observation outcome (counts)
    values = np.arange(N+1)
    # c : list of possible hidden states
    c = np.arange(m)

    ### Generate hidden state list
    data = np.array([])
    for j in range(l):
        # initialize each chain
        x_lst = np.array([], dtype='int32')
        initS = weighted_values(c, pi, 1)
        h_lst = initS
        # generate h and x
        for i in range(3):
            h_curr = h_lst[-1]
            q = p[h_lst[-1]]
            x = np.random.binomial(N, q)
            dist = T[h_curr]
            h_next = weighted_values(c, dist, 1)
            h_lst = np.append(h_lst, h_next)
            x_lst = np.append(x_lst, x)
        data = np.append(data, x_lst)

    ### Reshape data to l by 3
    data = np.reshape(data, [l, 3])
    return data

# <codecell>

def generateData_general(T, O, pi, l):
    ### This function generates l triples, each representing methylation counts at first 3 sites of a sequence
    ### Data generated from an HMM with transition matrix T and obervation matrix O
    #   T and O are column stochastic

    N = np.shape(O)[0]
    m = np.shape(O)[1]

    # create l triples (x1, x2, x3), x1, x2, x3 \in [N].
    T = np.transpose(T)
    O = np.transpose(O)
    # values : list of possible observation outcome (counts)
    values = np.arange(N+1)
    # p[] : methylation probability for hidden state
    # p = np.random.rand(m)
    # c : list of possible hidden states
    c = np.arange(m)


    ### Generate hidden state list
    data = np.array([])
    for j in range(l):
        # initialize each chain
        x_lst = np.array([], dtype='int32')
        initS = weighted_values(c, pi, 1)
        h_lst = initS
        # generate h and x
        for i in range(3):
            h_curr = h_lst[-1]
            obsDist = O[h_curr]
            x = weighted_values(values, obsDist, 1)
            dist = T[h_curr]
            h_next = weighted_values(c, dist, 1)
            h_lst = np.append(h_lst, h_next)
            x_lst = np.append(x_lst, x)
        data = np.append(data, x_lst)

    ### Reshape data to l by 3
    data = np.reshape(data, [l, 3])
    return data

# <codecell>
'''
# Make HMM
N = 3 #10
numObs = N+1
m = 2 #5
l = 100
min_sigma = 0.5 #0.1
T = makeTransitionMatrix(m, min_sigma)
O = makeObservationMatrix(m, numObs, min_sigma)
pi = makeDistribution(m)
Data = generateData_general(N, m, T, O, pi, l)

#Learn HMM
xRange = np.matrix(np.linspace(Data.min(),Data.max(),numObs)).T
import kernelNaiveBayes as knb
noise = np.random.rand(Data.shape[0],Data.shape[1])/1e3 #jiggle the data points a little bit to improve conditioning
O_hat = knb.kernXMM(Data+noise,2,xRange,var=0.5)

#Plot results for comparison
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(xRange,O_hat[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("est pdf");
ax = fig.add_subplot(1,2,2);
ax.plot(xRange,O_hat[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("est pdf");

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(xRange,O[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("true pdf");
ax = fig.add_subplot(1,2,2);
ax.plot(xRange,O[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("true pdf");
'''
