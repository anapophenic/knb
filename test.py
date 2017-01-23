import numpy as np
import scipy as sp
import binom_hmm as bh
import feature_map as fm
import matplotlib.pyplot as plt

'''
Test Script for computing the ground truth E[phi(x,t)|h]
TODO: extend this to full N distribution and make this modular
'''


if __name__ == '__main__':

    n = 40;
    N = 40;
    p = 0.0833;

    landmarks = bh.unif_partition(n);

    M = np.zeros((N+1,n));
    for i in range(N+1):
        M[i,:] = map(lambda t: fm.beta_interval(t, i, N-i, n), landmarks.tolist())
    #    for j in range(n):
    #        M[i,j] = sp.stats.beta.cdf(float(j+1)/n, i+1, N-i+1) - sp.stats.beta.cdf(float(j)/n, i+1, N-i+1)

    #print M[0,:]
    #

    prob = np.zeros(N+1);

    for i in range(N+1):
        prob[i] = sp.stats.binom.pmf(i, N, p)

    #print prob
    #print np.array(range(N+1)).dot(prob)

    C = prob.dot(M);

    fig = plt.figure(1)
    plt.plot(landmarks, C, color='r', linewidth=3)
    fig.savefig('temp.pdf')
    plt.close(fig)

    a = float(1)/(N+2)

    print C

    #print a
    #print np.sum(C, axis = 0)
    #print (C.dot(landmarks) - a) / (1 - 2*a)
    #print C.dot(landmarks)

    #means = M.dot(landmarks)
    #means_gt = np.array(map(lambda i: float(i+1)/(N+2),range(N+1)))

    #print float(N*p+1) / (N+2)
    #print (1 - 2*a) * p + a
    #print (means.dot(prob) - a) / (1 - 2*a)
    #print (means_gt.dot(prob) - a) / (1 - 2*a)
