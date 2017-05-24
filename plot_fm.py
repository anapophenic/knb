import feature_map as fm
import visualize as vis
import numpy as np
import utils as ut
import data_import as di

if __name__ == '__main__':

    ctxt_group = [range(12,16)]
    ce_group = ['E']
    s = 1
    ch = '1'

    coverage, methylated, N, x_zipped, a, p_c = di.data_prep_ctxt_ce(ctxt_group, ce_group, s, ch);

    p_c = p_c[0,:]
    N = 100
    p_c = p_c[:101]

    #cov_0 = coverage[0,:]
    #vis.print_hist(cov_0[cov_0 != 0], p_c)



    #for i in range(21,50):
    #N = 100
    #mu = i
    #p_c = ut.normalize_v(np.ones(N+1));

    #p_c = ut.truncated_poisson_pmf(mu, N)

    m = 20
    #p_h = ut.unif_partition(m);
    p_h = np.array([1/pow(1.5,i) for i in range(m)]);

    phi = fm.phi_beta_shifted_cached;

    n = 20

    C = fm.expected_fm_p_c(phi, n, p_c, p_h)

    cs = vis.default_color_scheme(m)

    vis.print_feature_map(C, cs, 'fm/', 'real_data' + 'p_h = ' + str(p_h[:3])+'.pdf', [0,20])
