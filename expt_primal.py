import moments_cons as mc
import numpy as np
import data_import as di
import matplotlib.pyplot as plt

if __name__ == '__main__':


    np.random.seed(0);
    #N = 3
    m = 3
    #play with n,
    #change the setting of t's, even forget about recovering p_h's
    #try to look at least squares soln
    n = 100
    
    '''
    N = 3
    #l = 20000
    min_sigma_t = 0.7
    #min_sigma_o = 0.5
    #n = 3;
    
    
    print 'Generating O and T..'
    #O = generate_O(m, N, min_sigma_o);
    p = generate_p(m);
    print 'p = '
    print p       
    
    #O = generate_O_binom(m, N, p);
    O = generate_O_stochastic_N(m, N, p);
    print 'O = '
    print O     
    
    T = generate_T(m, min_sigma_t);
    print 'T = '
    print T
    
    initDist = generate_initDist(m);
    print 'initDist = '
    print initDist
    
    a = get_a(N);
    '''
    
    #print 'Generating Data..'
    #X = dataGenerator.generateData_general(T, O, initDist, l)
    #X = dataGenerator.generateData_firstFew(N, m, T, p, initDist, l)
    chrs = [str(a) for a in range(1,20,1)]
    chrs.append('X')
    chrs.append('Y')
    
    cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    
    for ch in chrs:
        print 'ch = '
        print ch
        for ce in cells:
            print 'ce = '
            print ce
            for s in range(1,6):
                print 's = '
                print s
                for l in [10000, 20000, 40000, 80000, 160000, 320000]:
                    print 'l = '
                    print l
                    print 'Reading Data..'
                    filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_' + ce + '_chr' + ch + '_binsize100.mat'
                    N, X_importance_weighted, a = di.data_prep(filename, 'explicit',l,s);
                    print 'N = '
                    print N
                    print 'a = '
                    print a
                    #X = X[:10000,:]
                    
                    #phi = phi_onehot;
                    #phi = phi_beta;
                    phi = mc.phi_beta_shifted;
                    
                    if phi == mc.phi_onehot:
                        n = N + 1

                    #C = gt_obs(phi, N, n, O);

                    print 'Constructing Moments..'    
                    P_21, P_31, P_23, P_13, P_123 = mc.moments_cons_importance_weighted(X_importance_weighted, phi, N, n);
                    #P_21, P_31, P_23, P_13, P_123, C, S_1, S_3 = moments_gt(O, phi, N, n, T, initDist)
                    #R_21, R_31, R_23, R_13, R_123, C, S_1, S_3 = moments_gt(O, phi, N, n, T, initDist)
                    
                    #print 'C = '
                    #print C
                    
                    #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
                    #save the moments in a file
                    
                    print 'Estimating..'
                    
                    for m in range(2,10,1):
                        print 'm = '
                        print m
                        C_h, T_h = mc.estimate(P_21, P_31, P_23, P_13, P_123, m)
                        #C_h, T_h = estimate(R_21, R_31, R_23, R_13, R_123, m)
                        print 'C_h = '
                        print C_h
                        
                        print 'T_h = '
                        print T_h
                        
                        p_h = mc.get_p(phi, N, n, C_h, a)
                        print 'p_h = ' 
                        print p_h   
                        
                        fig = plt.figure(1)
                        plt.plot(mc.unif_partition(n), C_h)
                        fig.savefig('cell = ' + ce + 'chr = ' + ch + 'l = ' + str(l) + 's = ' + str(s) + '_m = ' + str(m) + '.pdf')   # save the figure to file
                        plt.close(fig)
                
                
                #print 'Refining using Binomial Knowledge'

                #C_h_p, T_h_p = estimate_refine(C_h, P_21, phi, N, n, m, a)
                #print 'C_h_p = '
                #print C_h_p
                #print 'T_h_p = '
                #print T_h_p
                #print get_p(phi, N, n, O_h)
               
