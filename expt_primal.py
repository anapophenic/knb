import moments_cons as mc
import data_generator as dg
import binom_hmm as bh
import feature_map as fm
import numpy as np
import data_import as di
import matplotlib.pyplot as plt
import hmm_inference as hi
import baum_welch as bw
import visualize as vis
import os
import sys
import itertools
import td_tpm
import td_als
import postprocess as pp
import utils as ut

def real_expt(phis, chrs, cell_groups, segments, lengths, lengths_test, n, ms, ctxt_groups, bw_iters, path_name, tex_name):

    #sys.stdout = open(path_name+'/parameters.txt', 'w+');
    vis.directory_setup(path_name);

    vis.print_doc_header(path_name, tex_name);

    for ch, ce_group, s, ctxt_group in itertools.product(chrs, cell_groups, segments, ctxt_groups):
        r = len(ctxt_group) * len(ce_group)
        print 'ch = \n' + str(ch) + '\nce_group = \n' + str(ce_group) + '\ns = \n' + str(s) + '\nctxt_group = \n' + str(ctxt_group)+ '\nr = \n' + str(r) + '\n'
        print 'Reading Data..'
        coverage, methylated, N, x_zipped, a = di.data_prep_ctxt_ce(ctxt_group, ce_group, s, ch);
        print 'r = ' + str(r)
        print '#rows of coverage (should be r)' + str(np.shape(coverage)[0])
        print 'coverage = \n' + str(coverage) + '\nmethylated = \n' + str(methylated) + '\n' #+ 'x_zipped = ' + x_zipped

        for l, l_test in itertools.product(lengths, lengths_test):
            print 'l = \n' + str(l) + '\nN = \n' + str(N) + '\na = \n' + str(a) + '\nl_test = \n' + str(l_test) + '\n'

            print 'Preparing training data..'
            coverage_train = coverage[:,:l]
            methylated_train = methylated[:,:l]
            x_importance_weighted = di.importance_weightify(x_zipped, l)

            print 'Preparing test data..'
            if l_test is None:
                l_test = np.shape(coverage)[1]
            coverage_test = coverage[:,:l_test].astype(float)
            methylated_test = methylated[:,:l_test].astype(float)
            coverage_test_reduced = np.sum(coverage_test, axis=0)
            idx = (coverage_test_reduced != 0)
            coverage_test_reduced = coverage_test[:,idx]
            methylated_test_reduced = methylated_test[:,idx]

            #vis.plot_m_and_c(coverage_test, methylated_test)

            for phi in phis:
                print 'phi = \n' + fm.phi_name(phi) + '\n'
                print 'Constructing Moments..'
                #P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(x_importance_weighted, phi, N, n);
                #vis.save_moments(P_21, P_31, P_23, P_13, P_123, ch, ce_group, s, ctxt_group, l, path_name)

                print 'Estimating..'
                sec_title = vis.get_sec_title(path_name, ce_group, ch, l, s, n, phi, ctxt_group)
                vis.print_expt_setting(path_name, sec_title, tex_name)
                vis.print_table_header(path_name, tex_name);


                td_algs = ['als']
                pp_algs = ['pos_als_iter']

                for m, td_alg, pp_alg in itertools.product(ms, td_algs, pp_algs):

                    pi_0 = ut.normalize_v(np.random.rand(m))
                    T_0 = ut.normalize_m(np.random.rand(m,m))
                    p_ch_0 = np.random.rand(r,m)
                    p_ch, T_h, pi_h = bw.baum_welch(coverage_train, methylated_train, p_ch_0, T_0, pi_0, 10)

                    '''
                    if td_alg == 'als':
                        C_h = td_als.als(P_123, m)
                    elif td_alg == 'tpm':
                        C_h = td_tpm.tpm(P_21, P_31, P_23, P_13, P_123, m)

                    C_h = pp.postprocess_m(C_h)

                    if pp_alg == 'pos':
                        C_h, T_h, pi_h = pp.refine_positify(C_h, P_21, P_31, P_23, P_13, P_123, m)
                    elif pp_alg == 'pos_als':
                        T_h, pi_h = pp.refine_nmf(P_21, C_h)
                    elif pp_alg == 'pos_als_iter':
                        C_h, T_h, pi_h = pp.refine_als_p21(P_21, C_h)

                    lims = fm.phi_lims(n, r);
                    p_ch = fm.get_pc(phi, N, C_h, a, lims)
                    '''

                    print 'm = \n' + str(m)
                    #print '\nC_h = \n' + str(C_h)
                    print '\nT_h = \n' + str(T_h) + '\npi_h = \n' + str(pi_h) + '\np_ch = \n' + str(p_ch) + '\n'
                    fig_title = vis.get_fig_title(ce_group, ch, l, s, m, n, phi, ctxt_group) + '_' + td_alg + '_' + pp_alg

                    p_x_h = lambda i: bh.p_x_ch_binom(p_ch, coverage_test, methylated_test, i)

                    print 'posterior decoding...'
                    h_dec_p = hi.posterior_decode(l_test, pi_h, T_h, p_x_h);
                    color_scheme = vis.get_color_scheme(h_dec_p, m)
                    posterior_title = fig_title + 'l_test = ' + str(l_test) + '_posterior.pdf'
                    bed_title = fig_title + 'l_test = ' + str(l_test) + '_bed'

                    h_dec_p = h_dec_p[idx]

                    #vis.browse_states(h_dec_p, path_name, posterior_title, color_scheme)
                    bed_list = vis.print_bed(h_dec_p, path_name, bed_title, m, ch, s)
                    bed_name = fig_title
                    vis.plot_meth_and_bed(coverage_test_reduced, methylated_test_reduced, bed_list, p_ch, bed_name, path_name, l, l_test)
                    #print 'viterbi decoding...'
                    #h_dec_v = hi.viterbi_decode(l_test, pi_h, T_h, p_x_h);
                    #viterbi_title = fig_title + 'l_test = ' + str(l_test) + '_viterbi.pdf'
                    #vis.browse_states(h_dec_v, viterbi_title, color_scheme)

                    #print 'generating feature map graph...'
                    #feature_map_title = fig_title + '_feature_map.pdf'
                    #vis.print_feature_map(C_h, color_scheme, path_name, feature_map_title, lims)

                    #print 'printing matrices'
                    print T_h
                    #T_title = fig_title + 'T_h.pdf'
                    #vis.print_m(T_h, T_title)

                    #print C_h
                    #C_title = fig_title + 'C_h.pdf'
                    #vis.print_m(C_h, C_title)

                    print p_ch
                    #p_title = fig_title + 'p_h.pdf'
                    #vis.print_v(p_h, p_title)

                    print pi_h
                    #pi_title = fig_title + 'pi_h.pdf'
                    #vis.print_v(pi_h, pi_title)

                    vis.print_fig_and(path_name, posterior_title,tex_name);
                    #vis.print_fig_bs(path_name, feature_map_title,tex_name);

                vis.print_table_aheader(path_name, tex_name);
    vis.print_doc_aheader(path_name, tex_name);


def synthetic_test_data(p, T, pi, N, l_test):

    p_N = dg.generate_p_N(N)
    O = bh.get_O_stochastic_N(p_N, p);
    x, h = dg.generate_seq(O, T, pi, l_test);
    browse_states(h)

    return x, h

def decoding_simulation(x, p_h, T_h, pi_h, N):
    l = len(x)
    coverage, methylated = zip(*map(lambda a: bh.to_c_m(a, N), x))

    p_x_h = lambda i: bh.p_x_h_binom(p_h, coverage, methylated, i)
    h_dec_p = hi.posterior_decode(l, pi_h, T_h, p_x_h);
    h_dec_v = hi.viterbi_decode(l, pi_h, T_h, p_x_h);

    browse_states(h_dec_p)
    browse_states(h_dec_v)
    #print zip(h, h_dec_p, h_dec_v)


def synthetic_expt(phi, path_name):

    try:
        os.stat(path_name)
    except:
        os.mkdir(path_name)

    #sys.stdout = open(path_name+'/parameters.txt', 'w+');

    n = 30
    N = 100
    l = 60000
    min_sigma_t = 0.8
    #min_sigma_o = 0.95
    #n = 3;
    #m = 3;

    #print 'Generating O and T..'
    #p = dg.generate_p(m);
    #print 'p = '
    #print p

    #p_N = dg.generate_p_N(N);
    #print 'p_N = '
    #print p_N

    #O = get_O_binom(m, N, p);
    #O = get_O(m, N, min_sigma_o);
    #O = bh.get_O_stochastic_N(p_N, p);
    #O = dg.generate_O(m, n, min_sigma_o);
    #print 'O = '
    #print O

    r = 2
    ms = [2 for i in range(r)]
    print ms
    m = ut.prod(ms)

    p_N = dg.generate_p_N(N);
    print 'p_N = '
    print p_N

    #p_ch = dg.generate_p_ch_random(ms);
    #p_ch = dg.generate_p_ch_cartesian(ms);
    p_ch = dg.generate_p_ch_monotone(m,r);
    print 'p_ch = '
    print p_ch

    T = dg.generate_T(m, min_sigma_t);
    print 'T = '
    print T

    pi = dg.generate_pi(m);
    print 'pi = '
    print pi

    #generate_O


    l_test = 500;
    #x_test, h_test = synthetic_test_data(p, T, pi, N, l_test);
    coverage_test, methylated_test, h_test = dg.generate_seq_bin_c(p_ch, p_N, T, pi, l_test);

    print 'Generating Data..'
    #x_zipped = dg.generate_longchain(T, O, pi, l)
    #x_zipped = dg.generate_firstfew(T, O, pi, l)
    #x_zipped = [tuple(row) for row in x_zipped]
    #X = dataGenerator.generateData_firstFew(N, m, T, p, pi, l)

    coverage, methylated, h = dg.generate_seq_bin_c(p_ch, p_N, T, pi, l);
    N, x_zipped, a = di.triples_from_seq(coverage, methylated, 'explicit')

    '''
    x_importance_weighted = di.importance_weightify(x_zipped, l);
    P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(x_importance_weighted, phi, N, n);
    '''
    #R_21, R_31, R_23, R_13, R_123, C, S_1, S_3 = mc.moments_gt(O, phi, N, n, T, pi)
    #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
    #print 'C = '
    #print C

    # ground truth states
    #vis.browse_states(h_test, path_name, 'gt l_test = ' + str(l_test) + '.pdf', color_scheme)
    bed_name_gts = 'gt_states'
    bed_list_gts = vis.print_bed(h_test, path_name, bed_name_gts, m, '1', 1)
    vis.plot_meth_and_bed(coverage_test, methylated_test, bed_list_gts, p_ch, bed_name_gts, path_name, l, l_test)

    # ground truth decoder
    #posterior_title = fig_title + 'l_test = ' + str(l_test) + '_posterior.pdf'
    #vis.browse_states(h_dec_p, path_name, posterior_title, color_scheme)
    #color_scheme = vis.get_color_scheme(h_dec_p, m)
    p_x_h = lambda i: bh.p_x_ch_binom(p_ch, coverage_test, methylated_test, i)
    h_dec_p = hi.posterior_decode(l_test, pi, T, p_x_h);
    bed_name_gtd = 'gt_decoder'
    bed_list_gtd = vis.print_bed(h_dec_p, path_name, bed_name_gtd, m, '1', 1)
    #vis.plot_meth_and_bed(coverage_test, methylated_test, bed_list_gtd, bed_name_gtd, path_name, l, l_test)
    vis.plot_meth_and_twobeds(coverage_test, methylated_test, bed_list_gts, p_ch, bed_list_gtd, p_ch, bed_name_gtd, path_name, l, l_test)


    for m_hat in range(4,5,1):

        '''
        #C_h = td_tpm.tpm(P_21, P_31, P_23, P_13, P_123, m_hat)
        C_h = td_als.als(P_123, m_hat)
        C_h = pp.postprocess_m(C_h)
        C_h, T_h, pi_h = pp.refine_positify(C_h, P_21, P_31, P_23, P_13, P_123, m_hat)
        #T_h, pi_h = pp.refine_nmf(P_21, C_h)
        lims = fm.phi_lims(n, r);
        p_ch_h = fm.get_pc(phi, N, C_h, a, lims)
        '''

        pi_0 = ut.normalize_v(np.random.rand(m))
        T_0 = ut.normalize_m(np.random.rand(m,m))
        p_ch_0 = np.random.rand(r,m)
        p_ch_h, T_h, pi_h = bw.baum_welch(coverage, methylated, p_ch_0, T_0, pi_0, 10)

        col_ind = ut.find_match(p_ch, p_ch_h)
        pi_h = pi_h[col_ind]
        p_ch_h = p_ch_h[:,col_ind]
        T_h = T_h[:, col_ind][col_ind, :]
        #C_h = C_h[:, col_ind]


        print T_h

        print 'm_hat = \n' + str(m)
        #print '\nC_h = \n' + str(C_h)
        print '\nT_h = \n' + str(T_h) + '\npi_h = \n' + str(pi_h) + '\np_ch_h = \n' + str(p_ch_h) + '\n'
        print 'p_ch_h - p_ch', np.linalg.norm(p_ch_h - p_ch)
        print 'T_h - T', np.linalg.norm(T_h - T)
        print 'pi_h - pi', np.linalg.norm(pi_h - pi)

        fig_title = 'synthetic'

        print 'posterior decoding...'

        # estimated decoder
        p_x_h_h = lambda i: bh.p_x_ch_binom(p_ch_h, coverage_test, methylated_test, i)
        h_dec_p_h = hi.posterior_decode(l_test, pi_h, T_h, p_x_h_h);
        posterior_h_title = fig_title + 'l_test = ' + str(l_test) + '_posterior_h.pdf'
        #vis.browse_states(h_dec_p, path_name, posterior_h_title, color_scheme)
        bed_name_ed = 'estimated_decoder'
        bed_list_ed = vis.print_bed(h_dec_p_h, path_name, bed_name_ed, m_hat, '1', 1)
        #vis.plot_meth_and_bed(coverage_test, methylated_test, bed_list_ed, bed_name_ed, path_name, l, l_test)
        vis.plot_meth_and_twobeds(coverage_test, methylated_test, bed_list_gts, p_ch, bed_list_ed, p_ch_h, bed_name_ed, path_name, l, l_test)

        #print 'viterbi decoding...'
        #h_dec_v = hi.viterbi_decode(l_test, pi_h, T_h, p_x_h);
        #viterbi_title = fig_title + 'l_test = ' + str(l_test) + '_viterbi.pdf'
        #vis.browse_states(h_dec_v, viterbi_title, color_scheme)

        print 'generating feature map graph...'
        color_scheme = vis.get_color_scheme(h_dec_p, m_hat)
        feature_map_title = fig_title + 'l = ' + str(l) + 'm_hat = ' + str(m_hat) + '_feature_map.pdf'
        #vis.print_feature_map(C_h, color_scheme, path_name, feature_map_title, lims)


        #print 'printing matrices'
        print 'T_h = '
        print T_h
        #T_title = fig_title + 'T_h.pdf'
        #vis.print_m(T_h, T_title)

        #print 'C_h\' = '
        #print C_h.T
        #C_title = fig_title + 'C_h.pdf'
        #vis.print_m(C_h, C_title)

        print 'p_ch_h = '
        print p_ch_h
        #p_title = fig_title + 'p_h.pdf'
        #vis.print_v(p_h, p_title)

        print 'pi_h = '
        print pi_h
        #pi_title = fig_title + 'pi_h.pdf'
        #vis.print_v(pi_h, pi_title)


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

    #path_name = 'synthetic/1026'
    #synthetic_expt(fm.phi_beta_shifted_cached, 3, path_name);


    #chrs = [str(a) for a in range(1,20,1)]
    #chrs.append('X')
    #chrs.append('Y')
    #chrs = ['1', '2']
    chrs = ['1']
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    #cells = ['E2', 'E1', 'E', 'V8', 'V9', 'V', 'P13P14', 'P15P16', 'P']
    #cells = ['E', 'V', 'P']
    #cell_groups = [['E', 'V']]
    cell_groups = [['E', 'V']]
    # n should be divisible by cell_groups * ctxt_groups
    n = 50
    ms = range(2, 10)
    #order: CC, CT, CA, CG
    #ctxt_groups = [[range(0,4)], [range(4,8)], [range(8,12)], [range(12,16)], [range(0,4), range(4,8), range(8,12), range(12, 16)]]
    #ctxt_groups = [[range(8,12), range(12,16)]]
    #ctxt_groups = [[range(0,4)]]
    #ctxt_groups = [[range(0,4), range(4,8), range(8,12), range(12, 16)]]
    ctxt_groups = [[range(12,16)]]

    '''
    Expt 1: Compare Binning Feature vs. Beta Feature

    '''

    path_name = '0522/'
    tex_name = 'result.tex'
    #segments = range(1, 6)
    #segments = range(1,5)
    segments = [1]
    lengths = [4000]
    #, 20000, 40000, 80000, 160000, 320000
    lengths_test = [10000]
    #phis = [mc.phi_beta_shifted_cached, mc.phi_binning_cached]
    #phis = [fm.phi_beta_shifted_cached]
    phis = [fm.phi_beta_shifted_cached_listify]
    real_expt(phis, chrs, cell_groups, segments, lengths, lengths_test, n, ms, ctxt_groups, 0, path_name, tex_name)


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

    '''
    Expt 3: Synthetic dataset
    '''
    '''
    phi = fm.phi_beta_shifted_cached_listify;
    path_name = 'synthetic/'

    synthetic_expt(phi, path_name)
    '''
