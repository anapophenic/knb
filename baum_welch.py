import numpy as np
import hmm_inference as hi
import binom_hmm as bh

#Baum-Welch algorithm for binomial HMMs
def baum_welch(coverage, methylated, p_0, T_0, pi_0, iters):

    l = np.shape(coverage)[0]
    nz_pos = coverage > 0
    nz_frac = np.array(methylated[nz_pos]) / np.array(coverage[nz_pos])

    pi = pi_0
    T = T_0
    p = p_0

    for t in range(iters):
        p_x_h = lambda i: bh.p_x_h_binom(p, coverage, methylated, i)
        alpha = hi.forward_var(l, pi, T, p_x_h)
        beta = hi.backward_var(l, pi, T, p_x_h)

        gamma, xi = expectation(alpha, beta, T, p_x_h)
        p, T, pi = maximization(gamma, xi, nz_frac, nz_pos)

        print 'p = '
        print p

        print 'T = '
        print T

        print 'pi = '
        print pi

    return p, T, pi

def maximization(gamma, xi, nz_frac, nz_pos):
    pi = gamma[:,0]

    m, l = np.shape(gamma)
    T = np.zeros((m, m))
    unn_T = np.sum(xi, axis=2);
    T = bh.normalize_m(unn_T);

    gammaf = gamma[:,nz_pos].dot(nz_frac);
    gammas = np.sum(gamma[:,nz_pos], axis=1);
    p = gammaf / gammas

    return p, T, pi

def expectation(alpha, beta, T, p_x_h):

    m, l = np.shape(alpha)
    gamma = np.zeros((m, l))
    xi = np.zeros((m, m, l-1));

    for i in range(l):
        # compute gamma
        unn_gamma = alpha[:,i] * beta[:,i]
        gamma[:,i] = bh.normalize_v(unn_gamma)

        # compute xi
        if i < l - 1:
            unn_xi = np.diag(beta[:,i+1] * p_x_h(i+1)).dot(T.dot(np.diag(alpha[:,i]))) ;
            xi[:,:,i] = bh.normalize_m_all(unn_xi)
            #print 'gamma_i = '
            #print gamma[:,i]
            #print 'marginal of xi ='
            #print np.sum(xi[:,:,i], axis=0)

    return gamma, xi
