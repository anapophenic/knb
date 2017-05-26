import moments_cons as mc
import data_generator as dg
import binom_hmm as bh
import feature_map as fm
import numpy as np
import data_import as di
import matplotlib.pyplot as plt
import hmm_inference as hi
import visualize as vis
import os
import sys
import itertools
import td_tpm
import td_als
import em_bmm as eb
import baum_welch as bw
import postprocess as pp
import utils as ut

class model:
    pass

def print_basic_info(mod):
    mod.r = len(mod.ctxt_group) * len(mod.ce_group)
    print ('ch = \n', mod.ch,
           '\nce_group = \n', mod.ce_group,
           '\ns = \n', mod.s,
           '\nctxt_group = \n', mod.ctxt_group,
           '\nr = \n', str(mod.r), '\n')

def print_data_info(mod):
    print ('#rows of coverage (should be r)', np.shape(mod.coverage)[0],
           'coverage = \n', mod.coverage,
           '\nmethylated = \n', mod.methylated,
           '\n') #+ 'x_zipped = ' + x_zipped

def print_data_selected_info(mod):
    print ('l = \n', mod.l,
           '\nN = \n', mod.N,
           '\na = \n', mod.a,
           '\nl_test = \n', mod.l_test,
           '\n')

def data_select(mod):
    print 'Preparing training data..'
    coverage_train = mod.coverage[:,:mod.l].astype(float)
    methylated_train = mod.methylated[:,:mod.l].astype(float)
    x_importance_weighted = di.importance_weightify(mod.x_zipped, mod.l)

    print 'Preparing test data..'
    if mod.l_test is None:
        mod.l_test = np.shape(mod.coverage)[1]
    coverage_test = mod.coverage[:,:mod.l_test].astype(float)
    methylated_test = mod.methylated[:,:mod.l_test].astype(float)

    return coverage_train, methylated_train, coverage_test, methylated_test, x_importance_weighted

def reduce_nonzero(mod):
    coverage_test_sum = np.sum(mod.coverage_test, axis=0)
    mod.nz_idx = (coverage_test_sum != 0)
    coverage_test_reduced = mod.coverage_test[:,mod.nz_idx]
    methylated_test_reduced = mod.methylated_test[:,mod.nz_idx]

    return coverage_test_reduced, methylated_test_reduced

def print_header(mod):
    mod.sec_title = vis.get_sec_title(mod)
    vis.print_expt_setting(mod)
    vis.print_table_header(mod)

def rand_init_params(m, r, n):
    pi_h = ut.normalize_v(np.random.rand(m))
    T_h = ut.normalize_m(np.random.rand(m,m))
    p_ch_h = np.random.rand(r,m)
    C_h = ut.normalize_m(np.random.rand(n,m))

    return p_ch_h, C_h, T_h, pi_h

def estimate_observation(mod):
    print 'Estimating..'
    p_ch_h, C_h, T_h, pi_h = rand_init_params(mod.m_h, mod.r, mod.n)
    if mod.td_alg == 'als':
        C_h = td_als.als(mod.P_123, mod.m_h)
    elif mod.td_alg == 'tpm':
        C_h = td_tpm.tpm(mod.P_21, mod.P_31, mod.P_23, mod.P_13, mod.P_123, mod.m_h)
    elif mod.td_alg == 'em_bmm':
        p_c = mod.p_c[:,:100]
        p_ch, pi_h = eb.em_bmm_group(mod.coverage_train, mod.methylated_train, p_ch, pi_h)
        C_h = fm.expected_fm_p_c_group(mod.phi, mod.n, p_c, p_ch)
    elif mod.td_alg == 'baum_welch':
        p_c = mod.p_c[:,:100]
        p_ch, T_h, pi_h = bw.baum_welch(mod.coverage_train, mod.methylated_train, p_ch_h, T_h, pi_h, 1)
        C_h = fm.expected_fm_p_c_group(mod.phi, mod.n, p_c, p_ch)

    return p_ch_h, C_h, T_h, pi_h

def postprocess(mod):
    mod.lims = fm.phi_lims(mod.n, mod.r);
    if mod.pp_alg == 'no':
        return mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h
    else:
        C_h = pp.postprocess_m(mod.C_h)
        if mod.pp_alg == 'pos':
            C_h, T_h, pi_h = pp.refine_positify(C_h, mod.P_21, mod.P_31, mod.P_23, mod.P_13, mod.P_123, mod.m)
        elif mod.pp_alg == 'pos_als':
            T_h, pi_h = pp.refine_nmf(mod.P_21, C_h)
        elif mod.pp_alg == 'pos_als_iter':
            C_h, T_h, pi_h = pp.refine_als_p21(mod.P_21, C_h)
        p_ch_h = fm.get_pc(mod.phi, mod.N, C_h, mod.a, mod.lims)
        return p_ch_h, C_h, T_h, pi_h

def print_params(mod):
    print 'm_h = \n' + str(mod.m_h)
    mod.color_scheme = vis.default_color_scheme(mod.m_h)
    #print '\nC_h = \n' + str(C_h)
    #print '\nT_h = \n' + str(mod.T_h) + '\npi_h = \n' + str(mod.pi_h) + '\np_ch = \n' + str(mod.p_ch_h) + '\n'

    print 'Printing matrices..'
    print mod.T_h
    T_title = mod.fig_title + 'T_h.pdf'
    vis.show_m(mod.T_h, T_title, mod.path_name, mod.state_name_h, True)

    print mod.C_h
    C_title = mod.fig_title + 'C_h.pdf'
    vis.show_m(mod.C_h, C_title, mod.path_name, mod.state_name_h, False)

    print mod.p_ch_h
    p_title = mod.fig_title + 'p_h.pdf'
    vis.show_m(mod.p_ch_h, p_title, mod.path_name, mod.state_name_h, False)

    print mod.pi_h
    pi_title = mod.fig_title + 'pi_h.pdf'
    vis.show_v(mod.pi_h, pi_title, mod.path_name, mod.state_name_h)

    print 'generating feature map graph...'
    mod.feature_map_title = mod.fig_title + '_feature_map.pdf'
    vis.print_feature_map(mod)

def print_decoding(coverage, methylated, h_dec_p, mod, option_str):
    mod.posterior_title = mod.fig_title + 'l_test = ' + str(mod.l_test) + option_str + '_posterior.pdf'
    mod.bed_title = mod.fig_title + 'l_test = ' + str(mod.l_test) + option_str + '_bed'

    bed_list = vis.print_bed(h_dec_p, mod)
    vis.plot_meth_and_bed(coverage, methylated, bed_list, mod)

def decode(mod):

    p_x_h = lambda i: bh.p_x_ch_binom(mod.p_ch_h, mod.coverage_test, mod.methylated_test, i)

    print 'posterior decoding...'
    mod.h_dec_p = hi.posterior_decode(mod.l_test, mod.pi_h, mod.T_h, p_x_h);

    print_decoding(mod.coverage_test, mod.methylated_test, mod.h_dec_p, mod, '')

    mod.coverage_test_reduced, mod.methylated_test_reduced = reduce_nonzero(mod)
    mod.h_dec_p_reduced = mod.h_dec_p[mod.nz_idx]
    print_decoding(mod.coverage_test_reduced, mod.methylated_test_reduced, mod.h_dec_p_reduced, mod, '_reduced_')

    #vis.browse_states(h_dec_p, path_name, posterior_title, color_scheme)

    #print 'viterbi decoding...'
    #h_dec_v = hi.viterbi_decode(l_test, pi_h, T_h, p_x_h);
    #viterbi_title = fig_title + 'l_test = ' + str(l_test) + '_viterbi.pdf'
    #vis.browse_states(h_dec_v, viterbi_title, color_scheme)



def real_expt(mod):
    #sys.stdout = open(path_name+'/parameters.txt', 'w+');
    vis.directory_setup(mod);
    vis.print_doc_header(mod);

    for mod.ch, mod.ce_group, mod.s, mod.ctxt_group in itertools.product(mod.chrs, mod.cell_groups, mod.segments, mod.ctxt_groups):
        print_basic_info(mod)
        mod.coverage, mod.methylated, mod.N, mod.x_zipped, mod.a, mod.p_c = di.data_prep_ctxt_ce(mod);
        print_data_info(mod)

        for mod.l, mod.l_test in itertools.product(mod.lengths, mod.lengths_test):
            print_data_selected_info(mod)
            mod.coverage_train, mod.methylated_train, mod.coverage_test, mod.methylated_test, mod.x_importance_weighted = data_select(mod)

            for mod.phi in mod.phis:
                mod.P_21, mod.P_31, mod.P_23, mod.P_13, mod.P_123 = mc.moments_cons_importance_weighted(mod);
                #vis.save_moments(P_21, P_31, P_23, P_13, P_123, ch, ce_group, s, ctxt_group, l, path_name)
                print_header(mod)

                for mod.m_h, (mod.td_alg, mod.pp_alg) in itertools.product(mod.ms, mod.selections):

                    mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h = estimate_observation(mod)
                    mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h = postprocess(mod)
                    mod.fig_title = vis.get_fig_title(mod)
                    mod.state_name_h = vis.state_name(mod.p_ch_h)

                    decode(mod)
                    print_params(mod)

                    vis.print_fig_and(mod)
                    vis.print_fig_bs(mod)

                vis.print_table_aheader(mod)
    vis.print_doc_aheader(mod)

# in synthetic data, we can have additional rows indicating ground truth states
# bed_name_gtd = 'gt_decoder'

def get_bed(mod, option_str, p_ch_h, h_seq):
    mod.bed_title = mod.fig_title + 'l_test = ' + str(mod.l_test) + option_str + '_bed'
    mod.m_h = np.shape(p_ch_h)[1]
    bed_list = vis.print_bed(h_seq, mod)
    state_name = vis.state_name(p_ch_h)
    return bed_list, state_name

def synthetic_print_decoding(mod, p_ch_h, T_h, pi_h, option_str):
    p_x_h = lambda i: bh.p_x_ch_binom(p_ch_h, mod.coverage_test, mod.methylated_test, i)
    h_dec_p = hi.posterior_decode(mod.l_test, pi_h, T_h, p_x_h);
    mod.bed_list_h, mod.state_name_h = get_bed(mod, option_str, p_ch_h, h_dec_p)
    vis.plot_meth_and_twobeds(mod.coverage_test, mod.methylated_test, mod)

def synthetic_matching(mod):
    print 'Matching..'
    col_ind = ut.find_match(mod.p_ch, mod.p_ch_h)
    pi_h = mod.pi_h[col_ind]
    p_ch_h = mod.p_ch_h[:,col_ind]
    T_h = mod.T_h[:, col_ind][col_ind, :]
    C_h = mod.C_h[:, col_ind]

    print 'm_hat = \n' + str(mod.m_h)
    print '\nC_h = \n' + str(C_h)
    print '\nT_h = \n' + str(T_h) + '\npi_h = \n' + str(pi_h) + '\np_ch_h = \n' + str(p_ch_h) + '\n'
    print '|p_ch_h - p_ch| = ', np.linalg.norm(p_ch_h - mod.p_ch)
    print '|T_h - T| = ', np.linalg.norm(T_h - mod.T)
    print '|pi_h - pi| = ', np.linalg.norm(pi_h - mod.pi)

    return p_ch_h, C_h, T_h, pi_h

def synthetic_expt(mod):

    vis.directory_setup(mod);
    #sys.stdout = open(path_name+'/parameters.txt', 'w+');

    mod.p_c, mod.p_ch, mod.T, mod.pi = generate_params(mod.N, mod.m, mod.r, mod.min_sigma_t)

    print 'Generating Data..'

    mod.coverage_test, mod.methylated_test, mod.h_test = dg.generate_seq_bin_c(mod, mod.l_test);
    mod.coverage_train, mod.methylated_train, mod.h_train = dg.generate_seq_bin_c(mod, mod.l);
    mod.N, mod.x_zipped = di.triples_from_seq(mod.coverage_train, mod.methylated_train, 'explicit')
    mod.a, mod.p_c_h = di.stats_from_seq(mod.coverage_train, mod.methylated_train)
    mod.x_importance_weighted = di.importance_weightify(mod.x_zipped, mod.l);
    mod.P_21, mod.P_31, mod.P_23, mod.P_13, mod.P_123 = mc.moments_cons_importance_weighted(mod);

    mod.fig_title = 'synthetic'

    mod.bed_list_gt, mod.state_name_gt = get_bed(mod, 'gt_state', mod.p_ch, mod.h_test)
    synthetic_print_decoding(mod, mod.p_ch, mod.T, mod.pi, 'gt_decoder')
    #R_21, R_31, R_23, R_13, R_123, C, S_1, S_3 = mc.moments_gt(O, phi, N, n, T, pi)
    #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
    #print 'C = '
    #print C

    for mod.m_h, (mod.td_alg, mod.pp_alg) in itertools.product(mod.ms, mod.selections):

        mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h = estimate_observation(mod)
        mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h = postprocess(mod)
        mod.p_ch_h, mod.C_h, mod.T_h, mod.pi_h = synthetic_matching(mod)

        print 'posterior decoding...'
        synthetic_print_decoding(mod, mod.p_ch_h, mod.T_h, mod.pi_h, 'estimated_decoder')
        print_params(mod)

def generate_params(N, m, r, min_sigma_t):
    p_c = dg.generate_p_c(N, r);
    print 'p_c = '
    print p_c

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

    return p_c, p_ch, T, pi


if __name__ == '__main__':

    np.random.seed(0);

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

    '''
    Expt 1: Compare Binning Feature vs. Beta Feature

    '''

    mod = model()
    #chrs = [str(a) for a in range(1,20,1)]
    #chrs.append('X')
    #chrs.append('Y')
    #chrs = ['1', '2']
    mod.chrs = ['1']
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    #cells = ['E2', 'E1', 'E', 'V8', 'V9', 'V', 'P13P14', 'P15P16', 'P']
    #cells = ['E', 'V', 'P']
    #cell_groups = [['E', 'V']]
    mod.cell_groups = [['E', 'V']]
    # n should be divisible by cell_groups * ctxt_groups
    mod.n = 50
    mod.ms = range(2, 10)
    #order: CC, CT, CA, CG
    #ctxt_groups = [[range(0,4)], [range(4,8)], [range(8,12)], [range(12,16)], [range(0,4), range(4,8), range(8,12), range(12, 16)]]
    #ctxt_groups = [[range(8,12), range(12,16)]]
    #ctxt_groups = [[range(0,4)]]
    #ctxt_groups = [[range(0,4), range(4,8), range(8,12), range(12, 16)]]
    mod.ctxt_groups = [[range(12,16)]]


    #'als', 'tpm',
    #td_algs = ['em_bmm', 'baum_welch']
    #pp_algs = ['pos', 'pos_als', 'pos_als_iter','no']

    mod.path_name = '0525/'
    mod.tex_name = 'result.tex'
    #segments = range(1, 6)
    #segments = range(1,5)
    mod.segments = [1]
    mod.lengths = [2000]
    #, 20000, 40000, 80000, 160000, 320000
    mod.lengths_test = [1000]
    #phis = [mc.phi_beta_shifted_cached, mc.phi_binning_cached]
    #phis = [fm.phi_beta_shifted_cached]
    mod.phis = [fm.phi_beta_shifted_cached_listify]
    mod.selections = [('baum_welch', 'no'), ('als', 'pos_als')]
    real_expt(mod)

    #real_expt(phis, chrs, cell_groups, segments, lengths, lengths_test, n, ms, ctxt_groups, 0, path_name, tex_name)

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
    mod = model()
    mod.ch = '1'
    mod.s = 1
    mod.phi = fm.phi_beta_shifted_cached_listify;
    mod.path_name = 'synthetic/'
    mod.n = 30
    mod.N = 100
    mod.l = 1000
    mod.l_test = 500;
    mod.min_sigma_t = 0.8
    mod.r = 2
    mod.m = 4
    mod.ms = range(4,5,1)
    mod.selections = [('baum_welch', 'no'), ('als', 'pos_als')]

    synthetic_expt(mod)
    '''
