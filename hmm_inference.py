import numpy as np
import data_import as di
from scipy import stats
import dataGenerator as dg
import moments_cons as mc

def forward_var(l, pi, T, p_x_h):
    m = np.shape(pi)[0];
    alpha = np.zeros((m, l));
    for i in range(l):
        if i == 0:
            alpha[:,i] = np.diag(p_x_h(i)).dot(pi);
        else:
            alpha[:,i] = np.diag(p_x_h(i)).dot(T.dot(alpha[:,i-1]))

    return alpha

def backward_var(l, pi, T, p_x_h):
    m = np.shape(pi)[0];
    beta = np.zeros((m, l));
    for i in range(l-1,-1,-1):
        if i == l-1:
            beta[:,i] = np.ones(m);
        else:
            beta[:,i] = (T.T).dot(np.diag(p_x_h(i)).dot(beta[:,i+1])).T

    return beta

def posterior_decode(l, pi, T, p_x_h):
    '''
    l: horizon
    pi: initial probability
    T: transition probability
    p_x_h: observation distribution
    '''

    m = np.shape(pi)[0];

    alpha = forward_var(l, pi, T, p_x_h);
    beta = backward_var(l, pi, T, p_x_h);

    h_dec = [];

    for i in range(l):
        alphabeta = alpha[:,i] * beta[:,i];
        h_dec.append(np.argmax(alphabeta))

    return h_dec

def p_x_h_binom(p_h, coverage, methylated, i):
    m = np.shape(p_h)[0];
    O_x = np.zeros(m);
    for j in range(m):
        O_x[j] = stats.binom.pmf(coverage, methylated, p[j])

    return O_x

def p_x_h_O(O, x, i):
    return O[x[i],:]

if __name__ == '__main__':

    #Synthetic Dataset:
    n = 20
    m = 6
    l = 50
    min_sigma_t = 0.7
    min_sigma_o = 0.8

    O = dg.generate_O(m, n, min_sigma_o);
    print 'O = '
    print O

    T = dg.generate_T(m, min_sigma_t);
    print 'T = '
    print T

    pi = dg.generate_pi(m);
    print 'pi = '
    print pi

    x, h = dg.generate_seq(T, O, pi, l);
    p_x_h = lambda i: p_x_h_O(O, x, i);

    h_dec = posterior_decode(l, pi, T, p_x_h);

    print h
    print h_dec


    #Real Dataset:
    #chrs = [str(a) for a in range(1,20,1)]
    #chrs.append('X')
    #chrs.append('Y')
    #chrs = ['1']
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    #cells = ['E2', 'E1', 'E']

    #ch = chrs[0];
    #ce = cells[0];

    #filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_' + ce + '_chr' + ch + '_binsize100.mat';

    #ctxt = range(12, 16)
    #segments = range(1, 6)
    #s = segments[0];

    #lengths = [320000]
    #l = lengths[0];

    #(coverage, methylated) = seq_prep(filename, l, s, ctxt);
