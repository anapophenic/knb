import data_import as di
import data_generator as dg
import moments_cons as mc
import binom_hmm as bh
import numpy as np
from scipy import stats

'''
Throughout we use the following conventions:
l: length of the horizon
pi: initial probability
T: transition probability
p_x_h: observation distribution (in functional form)
'''

def forward_var(l, pi, T, p_x_h):
    m = np.shape(pi)[0];
    alpha = np.zeros((m, l));
    ln_z = np.zeros(l);

    for i in range(l):
        if i == 0:
            alpha[:,i] = np.diag(p_x_h(i)).dot(pi);
            ln_z[i] = np.log(np.sum(alpha[:,i]));
        else:
            alpha[:,i] = np.diag(p_x_h(i)).dot(T.dot(alpha[:,i-1]));
            ln_z[i] = ln_z[i-1] + np.log(np.sum(alpha[:,i]));

        #normalize(for numerical stability)
        alpha[:,i] = alpha[:,i] / np.sum(alpha[:,i])
        if ln_z[i] == np.nan:
            break
        #print 'alpha[:,i] = '
        #print alpha[:,i]

    return alpha, ln_z

def backward_var(l, pi, T, p_x_h):
    m = np.shape(pi)[0];
    beta = np.zeros((m, l));
    ln_z = np.zeros(l);

    for i in range(l-1,-1,-1):
        if i == l-1:
            beta[:,i] = np.ones(m);
            ln_z[i] = np.log(np.sum(beta[:,i]));
        else:
            beta[:,i] = (T.T).dot(np.diag(p_x_h(i+1)).dot(beta[:,i+1])).T
            ln_z[i] = ln_z[i+1] + np.log(np.sum(beta[:,i]))
        #normalize(for numerical stability)
        beta[:,i] = beta[:,i] / np.sum(beta[:,i])
        if ln_z[i] == np.nan:
            break
        #print 'alpha[:,i] = '
        #print alpha[:,i]

    return beta, ln_z

def posterior_decode(l, pi, T, p_x_h):

    m = np.shape(pi)[0];

    alpha, ln_z_alpha = forward_var(l, pi, T, p_x_h);
    beta, ln_z_beta = backward_var(l, pi, T, p_x_h);

    h_dec = np.int_(np.zeros(l));

    for i in range(l):
        # Sanity Check:
        # The likelihood of the data wrt the model should be independent of the choice of i
        # print ln_z_alpha[i] + ln_z_beta[i] + np.log(alpha[:,i].dot(beta[:,i]))
        alphabeta = alpha[:,i] * beta[:,i];
        h_dec[i] = np.argmax(alphabeta)

    return h_dec

def viterbi_decode(l, pi, T, p_x_h):

    m = np.shape(pi)[0];
    gamma = np.zeros((m,l));
    pre = np.int_(np.zeros((m,l)));
    h_dec = np.int_(np.zeros(l));

    for i in range(l):
        if i == 0:
            for j in range(m):
                gamma[j,0] = np.log(pi[j]) + np.log(p_x_h(i)[j])
        else:
            for j in range(m):
                max_k = 0;
                max_lkhd = -np.inf;
                for k in range(m):
                    tmp_lkhd = np.log(p_x_h(i)[j]) + np.log(T[j,k]) + gamma[k,i-1]
                    if tmp_lkhd > max_lkhd:
                        max_k = k;
                        max_lkhd = tmp_lkhd;
                pre[j,i] = max_k;
                gamma[j,i] = max_lkhd;

    max_j = 0
    max_lkhd = -np.inf
    for j in range(m):
        if gamma[j,l-1] > max_lkhd:
            max_j = j
            max_lkhd = gamma[j,l-1];

    h_dec[l-1] = max_j;

    for i in range(l-2,-1,-1):
        max_j = pre[max_j,i+1]
        h_dec[i] = max_j;

    return h_dec

if __name__ == '__main__':

    #Synthetic Dataset:
    n = 20
    m = 6
    l = 100
    N = 20
    min_sigma_t = 0.6
    min_sigma_o = 0.7
    r = 4

    #O = dg.generate_O(m, n, min_sigma_o);
    #print 'O = '
    #print O

    print 'Generating O and T..'
    #p = dg.generate_p(m);
    #print 'p = '
    #print p

    p_N = dg.generate_p_N(N);
    print 'p_N = '
    print p_N

    p_ch = dg.generate_p_ch(m, r);
    print 'p_ch = '
    print p_ch

    #O = get_O_binom(m, N, p);
    #O = get_O(m, N, min_sigma_o);
    #O = bh.get_O_stochastic_N(p_N, p);
    #O = dg.generate_O(m, n, min_sigma_o);
    #print 'O = '
    #print O

    T = dg.generate_T(m, min_sigma_t);
    print 'T = '
    print T

    pi = dg.generate_pi(m);
    print 'pi = '
    print pi

    #x, h = dg.generate_seq(O, T, pi, l);
    #p_x_h = lambda i: bh.p_x_h_O(O, x, i);

    coverage, methylated, h = dg.generate_seq_bin_c(p_ch, p_N, T, pi, l)
    p_x_h = lambda i: bh.p_x_ch_binom(p_ch, coverage, methylated, i)

    h_dec_p = posterior_decode(l, pi, T, p_x_h);
    h_dec_v = viterbi_decode(l, pi, T, p_x_h);

    print h
    print h_dec_p
    print h_dec_v

    #(coverage, methylated) = seq_prep(filename, l, s, ctxt);


    '''
    #Real Dataset:
    #chrs = [str(a) for a in range(1,20,1)]
    #chrs.append('X')
    #chrs.append('Y')
    chrs = ['1']
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    cells = ['E2', 'E1', 'E']

    ch = chrs[0];
    ce = cells[0];

    filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_' + ce + '_chr' + ch + '_binsize100.mat';

    #ctxt = range(16)
    ctxt = range(12, 16)
    segments = range(1, 6)
    s = segments[0];

    lengths = [320000]
    l = lengths[0];

    phi = mc.phi_beta_shifted_cached;
    n = 20
    m = 4

    N, X_zipped, a = di.data_prep(filename,'explicit', None, s, ctxt);
    X_importance_weighted = di.prefix(X_zipped, l)
    P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(X_importance_weighted, phi, N, n);
    C_h, T_h, pi_h = mc.estimate(P_21, P_31, P_23, P_13, P_123, m)
    p_h = mc.get_p(phi, N, n, C_h, a)

    #T_h = np.array([[ 0.8993886, 0.28601558],[0.1006114 , 0.71398442]])
    #pi_h = np.array([ 0.73962056, 0.26037944])
    #p_h = np.array([ 0.92750509,  0.30209752])

    print T_h
    print pi_h
    print p_h

    l_i = 320000
    coverage, methylated = di.seq_prep(filename, l_i, s, ctxt);

    p_x_h = lambda i: bh.p_x_h_binom(p_h, coverage, methylated, i)
    h_dec_p = posterior_decode(l_i, pi_h, T_h, p_x_h);

    print h_dec_p.tolist()
    '''
