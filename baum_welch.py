import numpy as np
import hmm_inference as hi
import binom_hmm as bh
import utils as ut

#Baum-Welch algorithm for binomial HMMs
def baum_welch(coverage, methylated, p_0, T_0, pi_0, iters=10):

    r, l = np.shape(coverage)

    pi = pi_0
    T = T_0
    p = p_0

    for t in range(iters):
        p_x_h = lambda i: bh.p_x_ch_binom(p, coverage, methylated, i)
        alpha, ln_z = hi.forward_var(l, pi, T, p_x_h)
        beta, ln_z = hi.backward_var(l, pi, T, p_x_h)

        gamma, xi = expectation(alpha, beta, T, p_x_h)
        p, T, pi = maximization(gamma, xi, coverage, methylated)

        print 'p = \n', p
        print 'T = \n', T
        print 'pi = \n', pi

    return p, T, pi

def maximization(gamma, xi, coverage, methylated):
    # pi
    pi = np.sum(gamma,axis=1)
    pi = ut.normalize_v(pi)

    # T
    m, l = np.shape(gamma)
    T = np.zeros((m, m))
    unn_T = np.sum(xi, axis=2);
    T = ut.normalize_m(unn_T);

    # p
    met_all = methylated.dot(gamma.T);
    cov_all = coverage.dot(gamma.T);
    p = met_all / cov_all

    return p, T, pi

def expectation(alpha, beta, T, p_x_h):
    print np.shape(alpha)
    m, l = np.shape(alpha)
    gamma = np.zeros((m, l))
    xi = np.zeros((m, m, l-1));

    for i in range(l):
        # compute gamma
        unn_gamma = alpha[:,i] * beta[:,i]
        gamma[:,i] = ut.normalize_v(unn_gamma)

        # compute xi
        if i < l - 1:
            unn_xi = np.diag(beta[:,i+1] * p_x_h(i+1)).dot(T.dot(np.diag(alpha[:,i]))) ;
            xi[:,:,i] = ut.row_col_normalize_l1(unn_xi)
            #print 'gamma_i = '
            #print gamma[:,i]
            #print 'marginal of xi ='
            #print np.sum(xi[:,:,i], axis=0)

    return gamma, xi
