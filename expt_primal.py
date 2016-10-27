import moments_cons as mc
import data_generator as dg
import binom_hmm as bh
import feature_map as fm
import numpy as np
import data_import as di
import matplotlib.pyplot as plt
import hmm_inference as hi
import os
import sys

def real_expt(phis, chrs, cells, segments, lengths, n, ms, ctxt, path_name):

    try:
        os.stat(path_name)
    except:
        os.mkdir(path_name)

    sys.stdout = open(path_name+'/parameters.txt', 'w+');

    for ch in chrs:
        print 'ch = '
        print ch
        for ce in cells:
            print 'ce = '
            print ce
            for s in segments:
                print 's = '
                print s
                print 'Reading Data..'
                filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_' + ce + '_chr' + ch + '_binsize100.mat'
                N, x_zipped, a = di.data_prep(filename,'explicit', None, s, ctxt);

                #for l in [10000, 20000, 40000, 80000, 160000, 320000]:
                for l in lengths:
                    print 'l = '
                    print l
                    print 'N = '
                    print N
                    print 'a = '
                    print a

                    x_importance_weighted = di.importance_weightify(x_zipped, l)
                    print l

                    #X = X[:10000,:]

                    for phi in phis:
                        print 'phi = '
                        print phi_name(phi)

                        print 'Constructing Moments..'
                        P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(X_importance_weighted, phi, N, n);

                        #print 'C = '
                        #print C

                        #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
                        #save the moments in a file

                        print 'Estimating..'

                        for m in ms:
                            print 'm = '
                            print m
                            C_h, T_h, pi_h = mc.estimate(P_21, P_31, P_23, P_13, P_123, m)
                            #C_h, T_h, pi_h = estimate(R_21, R_31, R_23, R_13, R_123, m)
                            print 'C_h = '
                            print C_h

                            print 'T_h = '
                            print T_h

                            p_h = fm.get_p(phi, N, n, C_h, a)
                            print 'p_h = '
                            print p_h

                            fig = plt.figure(1)
                            ax = fig.add_subplot(1,1,1)
                            ax.set_title(r'Expected Feature Map given Hidden States')

                            ax.set_xlabel(r'$t$')
                            ax.set_ylabel(r'$\mathbb{E}[\phi(x,t)|h]$')

                            plt.plot(bh.unif_partition(n), C_h, linewidth=3)

                            fig.savefig(path_name + '/' + 'cell = ' + ce + '_chr = ' + ch + '_l = ' + str(l) + '_s = ' + str(s) + '_m = ' + str(m) + '_n = ' + str(n) + '_phi = ' + fm.phi_name(phi) + '_ctxt = ' + ctxt_name(ctxt) + '.pdf')
                            # save the figure to file
                            plt.close(fig)
                            #print 'Refining using Binomial Knowledge'

                            #C_h_p, T_h_p = estimate_refine(C_h, P_21, phi, N, n, m, a)
                            #print 'C_h_p = '
                            #print C_h_p
                            #print 'T_h_p = '
                            #print T_h_p
                            #print get_p(phi, N, n, O_h)

def synthetic_test_data(p, T, pi, N, l_test):

    p_N = dg.generate_p_N(N)
    O = bh.get_O_stochastic_N(p_N, p);
    x, h = dg.generate_seq(O, T, pi, l_test);
    browse_states(h)

    return x, h;

def decoding_simulation(x, p_h, T_h, pi_h, N):
    l = len(x)
    coverage, methylated = zip(*map(lambda a: fm.to_c_m(a, N), x))

    p_x_h = lambda i: bh.p_x_h_binom(p_h, coverage, methylated, i)
    h_dec_p = hi.posterior_decode(l, pi_h, T_h, p_x_h);
    h_dec_v = hi.viterbi_decode(l, pi_h, T_h, p_x_h);

    browse_states(h_dec_p)
    browse_states(h_dec_v)
    #print zip(h, h_dec_p, h_dec_v)

def browse_states(h):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    plt.figure()
    i_total = max(h) + 1
    l = len(h);
    plt.hold(True)
    #fig, axes = plt.subplots(i_total, 1)
    for i in range(i_total):
        plt.bar(xrange(l), [h[j] == i for j in xrange(l)], 1, color=colors[i], edgecolor='none')

    plt.hold(False)
    #for ax, i in zip(axes, xrange(l)):

    plt.show()

def synthetic_expt(phi, m, path_name):

    try:
        os.stat(path_name)
    except:
        os.mkdir(path_name)

    #sys.stdout = open(path_name+'/parameters.txt', 'w+');

    n = 20
    N = 30
    l = 500
    min_sigma_t = 0.7
    min_sigma_o = 0.9
    #n = 3;
    #m = 3;

    print 'Generating O and T..'
    p = dg.generate_p(m);
    print 'p = '
    print p

    p_N = dg.generate_p_N(N);
    print 'p_N = '
    print p_N

    #O = get_O_binom(m, N, p);
    #O = get_O(m, N, min_sigma_o);
    O = bh.get_O_stochastic_N(p_N, p);
    #O = dg.generate_O(m, n, min_sigma_o);
    print 'O = '
    print O

    T = dg.generate_T(m, min_sigma_t);
    print 'T = '
    print T

    pi = dg.generate_pi(m);
    print 'pi = '
    print pi

    l_test = 500;
    x_test, h_test = synthetic_test_data(p, T, pi, N, l_test);

    print 'Generating Data..'
    #x_zipped = dg.generate_longchain(T, O, pi, l)
    x_zipped = dg.generate_firstfew(T, O, pi, l)
    #x_zipped = [tuple(row) for row in x_zipped]
    #X = dataGenerator.generateData_firstFew(N, m, T, p, pi, l)

    x_importance_weighted = di.importance_weightify(x_zipped, l);
    P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(x_importance_weighted, phi, N, n);
    R_21, R_31, R_23, R_13, R_123, C, S_1, S_3 = mc.moments_gt(O, phi, N, n, T, pi)
    #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
    print 'C = '
    print C

    for m_hat in range(2,4,1):
        print 'm_hat = '
        print m_hat
        C_h, T_h, pi_h = mc.estimate(P_21, P_31, P_23, P_13, P_123, m_hat)
        #C_h, T_h, pi_h = mc.estimate(R_21, R_31, R_23, R_13, R_123, m_hat)
        print 'C_h = '
        print C_h

        p_h = fm.get_p(phi, N, n, C_h, fm.get_a(N))
        print 'p_h = '
        print p_h

        print 'T_h = '
        print T_h

        print 'pi_h = '
        print pi_h

        fig = plt.figure(1)
        plt.plot(bh.unif_partition(n), C_h)
        fig.savefig( path_name + '/' + 'phi = ' + fm.phi_name(phi) + '_m = ' + str(m) +  '_l = ' + str(l) + '_m_hat = ' + str(m_hat) + '_n = ' + str(n) + '.pdf')   # save the figure to file
        plt.close(fig)

        decoding_simulation(x_test, p_h, T_h, pi_h, N)

    raw_input()


if __name__ == '__main__':


    np.random.seed(0);

    #N = 3
    #m = 3
    #play with n,
    #change the setting of t's, even forget about recovering p_h's
    #try to look at least squares soln


    # Todos May 30:
    # Finish up Binnning Feature Map DONE
    # Find some memory-efficient way to perform calculation (as opposed to calling beta integral each time) DONE
    # Figure out some ways to speed up tensor matrix multiplication
    #

    # Todos June 2:
    # Compute the marginal of coverage, then use generate_O_stochastic_N in the refined recoverage.

    # Todos June 14:
    # re-run all expts with E2 (all configs)
    # merge the same cell types to get a big dataset (me1 + me2 / cov1 + cov2)

    #phi = phi_onehot;
    #phi = phi_beta;
    #phi = mc.phi_beta_shifted_cached;
    #phi = mc.phi_binning_cached;

    #if phi == mc.phi_onehot:
    #    n = N + 1

    #for phi in [fm.phi_beta_shifted_cached, fm.phi_binning_cached]:
    #    for m in range(2,10,1):
    #        synthetic_expt(phi, m)

    path_name = 'synthetic/1026'
    synthetic_expt(fm.phi_beta_shifted_cached, 3, path_name);

    '''
    #chrs = [str(a) for a in range(1,20,1)]
    #chrs.append('X')
    #chrs.append('Y')
    chrs = ['1']
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    cells = ['E2', 'E1', 'E']
    n = 50
    ms = range(1, 9)
    ctxt = range(12, 16)
    '''

    '''
    Expt 1: Compare Binning Feature vs. Beta Feature

    '''
    '''
    path_name = 'cg'
    segments = range(1, 6)
    lengths = [320000]
    phis = [mc.phi_beta_shifted_cached, mc.phi_binning_cached]
    real_expt(phis, chrs, cells, segments, lengths, n, ms, ctxt, path_name)
    '''

    '''
    Expt 2: Vary the number of Segments

    '''
    '''

    path_name = 'vary_s'
    segments = range(1,6)
    lengths = [320000]
    phis = [mc.phi_beta_shifted_cached]
    real_expt(phis, chrs, cells, segments, lengths, n, ms, ctxt, path_name)
    '''
