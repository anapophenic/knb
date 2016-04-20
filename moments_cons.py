import dataGenerator
import data_import
import numpy as np
from scipy import stats
from scipy import special
import tentopy
import kernelNaiveBayes

def moments_cons(X, phi, N, n):
    
    P_21 = np.zeros((n,n));
    P_31 = np.zeros((n,n));    
    P_23 = np.zeros((n,n));    
    P_13 = np.zeros((n,n));        
    P_123 = np.zeros((n,n,n));         
    
    l = np.shape(X)[0];
    s = float(l);
    
    for i in xrange(l):
        print i
        p1 = phi(X[i,0], N, n);
        p2 = phi(X[i,1], N, n);
        p3 = phi(X[i,2], N, n);
    
        P_21 += np.kron(p2, p1). reshape(n, n) / s
        P_31 += np.kron(p3, p1). reshape(n, n) / s
        P_23 += np.kron(p2, p3). reshape(n, n) / s
        P_13 += np.kron(p1, p3). reshape(n, n) / s
        P_123 += np.kron(p1, np.kron(p2, p3)). reshape(n, n, n) / s
        
    return P_21, P_31, P_23, P_13, P_123
    
    
def moments_cons_importance_weighted(X_iw, phi, N, n):
    
    P_21 = np.zeros((n,n));
    P_31 = np.zeros((n,n));    
    P_23 = np.zeros((n,n));    
    P_13 = np.zeros((n,n));        
    P_123 = np.zeros((n,n,n));         
    
    s = sum(X_iw.values());
    print len(X_iw)
    
    
    i = 0
    for X, importance_weight in X_iw.iteritems():
        #if importance_weight > 1:
        #    print X, importance_weight
        if i % 10000 == 0:
            print i
        i += 1
        p1 = phi(X[0], N, n);
        p2 = phi(X[1], N, n);
        p3 = phi(X[2], N, n);
    
        P_21 += importance_weight * np.kron(p2, p1). reshape(n, n) / s
        P_31 += importance_weight * np.kron(p3, p1). reshape(n, n) / s
        P_23 += importance_weight * np.kron(p2, p3). reshape(n, n) / s
        P_13 += importance_weight * np.kron(p1, p3). reshape(n, n) / s
        P_123 += importance_weight * np.kron(p1, np.kron(p2, p3)). reshape(n, n, n) / s
        
    return P_21, P_31, P_23, P_13, P_123

def moments_gt(O, phi, N, n, T, initDist):
    '''
    P_21 = C * T * diag(initDist) * C.T
    P_31 = C * T * T * diag(initDist) * C.T
    
    P_32 = C * T * diag(T * initDist) * C.T
    P_23 = P_32.T
    P_13 = P_31.T
    
    C_1 = C * diag(initDist) * T.T * diag(T*initDist)^{-1} 
    C_2 = C
    C_3 = C * T
    
    P_123 = I (C_1, C_2 * diag(T * initDist), C_3)
    
    
    P_21 = np.dot(C, np.dot(T, np.dot(np.diag(initDist), C.T)))
    P_31 = np.dot(C, np.dot(T, np.dot(T, np.dot(np.diag(initDist), C.T))))
    P_32 = np.dot(C, np.dot(T, np.dot(np.diag(np.dot(T,initDist)), C.T)))
    P_23 = P_32.T
    P_13 = P_31.T
    
    C_1 = np.dot(C, np.dot(np.diag(initDist), np.dot(T.T, np.linalg.inv(np.diag(np.dot(T, initDist))))))
    C_1_tilde = np.dot(C, np.dot(np.diag(initDist), T.T))
    C_2 = C
    C_3 = np.dot(C, T)
    P_123 = kernelNaiveBayes.trilinear('I', C_1.T, np.dot(C_2, np.diag(np.dot(T, initDist))).T, C_3.T);    
    '''
 
    C = gt_obs(phi, N, n, O);
   
    R_21 = C.dot(T.dot(np.diag(initDist).dot(C.T)))
    R_31 = C.dot(T.dot(T.dot(np.diag(initDist).dot(C.T))))
    R_32 = C.dot(T.dot(np.diag(T.dot(initDist)).dot(C.T)))
    R_23 = R_32.T
    R_13 = R_31.T
    
    C_1 = C.dot(np.diag(initDist).dot(T.T.dot(np.linalg.inv(np.diag(T.dot(initDist))))))
    C_2 = C
    C_3 = C.dot(T)
    
    R_123 = kernelNaiveBayes.trilinear('I', C_1.T, C_2.dot(np.diag(T.dot(initDist))).T, C_3.T);
    
    S_1 = C_2.dot(np.linalg.pinv(C_1))
    S_3 = C_2.dot(np.linalg.pinv(C_3))

    return R_21, R_31, R_23, R_13, R_123, C, S_1, S_3


def range_cons(P, m):
    #print P
    #np.random.randn(P.shape[0], P.shape[1])
    U, S, V = np.linalg.svd(P, full_matrices=False)    
    #print S
    #print U[:,:m].shape
    return U[:,:m]
    

def symmetrize(P_21, P_31, P_23, P_13, P_123, U):

    Q_21 = U.T.dot(P_21.dot(U));
    Q_31 = U.T.dot(P_31.dot(U));
    Q_23 = U.T.dot(P_23.dot(U));
    Q_13 = U.T.dot(P_13.dot(U));
    
    S_1 = Q_23.dot(np.linalg.inv(Q_13));
    S_3 = Q_21.dot(np.linalg.inv(Q_31));
    
    M_2 = S_1.dot(Q_21.T);
    M_2 = (M_2 + M_2.T)/2;
    
    #print np.shape(P_123)
    #Q_123 = np.tensordot(P_123, U, axes=([0], [0]));
    #print np.shape(Q_123)
    #Q_123 = np.tensordot(Q_123, U, axes=([2], [0]));  
    #M_3 = np.tensordot(Q_123, S_1, axes=([0], [1]));
    #M_3 = np.tensordot(M_3, S_3, axes=([2], [1]));

    M_3 = kernelNaiveBayes.trilinear(P_123, U.dot(S_1.T), U, U.dot(S_3.T))

    return M_2, M_3

def phi_beta_shifted(x, N, n):
    '''
        Input: 
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map  
        Output:
            beta-distribution encoding of phi(x)
    '''
    #print x
    #print N
    i = int(x / (N+1));
    k = int(x) % (N+1); 
    if k > i:
        return np.zeros(n)
    
    #p = np.asarray(map(lambda t: (t ** k) * ( (1-t) ** (i - k) ), unif_partition(n).tolist()));
    p = np.asarray(map(lambda t: beta_interval(t, k, i-k, n), unif_partition(n).tolist()));
    
    return p / sum(p);


def phi_beta(x, N, n):
    '''
        Input: 
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map  
        Output:
            beta-distribution encoding of phi(x)
    '''
    #print x
    #print N
    
    #p = np.asarray(map(lambda t: (t ** x) * ( (1-t) ** (N-x) ), unif_partition(n).tolist()));
    p = np.asarray(map(lambda t: beta_interval(t, x, N-x, n), unif_partition(n).tolist()));
    return p / sum(p);
    
def phi_onehot(x, N, n):
    '''
        Input: 
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map  
        Output:
            one hot encoding of phi(x)
    '''
    p = np.zeros(n);
    p[int(x)] = 1
    return p;
    
def gt_obs(phi, N, n, O):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map  
            O: one hot observation matrix
        Output:
            O_m: feature obseravtion matrix
    '''
    
    if ((phi == phi_onehot) or (phi == phi_beta)):
        Trans = np.zeros((n, N+1));
        for x in xrange(N+1):
            Trans[:,x] = phi(x, N, n).T
    elif phi == phi_beta_shifted:
        Trans = np.zeros((n, (N+1)*(N+1)));
        for x in xrange((N+1)*(N+1)):
            Trans[:,x] = phi(x, N, n).T
    
    #print 'Trans = '
    #print Trans
    #print 'O = '
    #print O    
        
    O_m = Trans.dot(O);
    return O_m;

def get_a(N):
    return sum(map(lambda n: 1.0/(n+2), range(0,N+1,1))) / (N+1)
    
def beta_interval(t, k, l, n):    
    return stats.beta.cdf(t+0.5/n, k+1, l+1) - stats.beta.cdf(t-0.5/n, k+1, l+1)
    
def unif_partition(n):
    return np.linspace(1.0/(2*n), 1.0 - 1.0/(2*n), n)
    
def get_O(phi, N, n, C_h, a):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map
            C_h: estimated observation matrix
        Output:
            p_h: estimated methylating probability
    '''
    
    if (phi == phi_onehot):
        p_h = np.sum(np.diag(np.linspace(0,N,N+1)).dot(C_h), axis = 0) / N
        O_h = generate_O_binom(m, N, p_h)
    elif (phi == phi_beta):
        p_h = ((N+1) * np.sum(np.diag(unif_partition(n)).dot(C_h), axis = 0) - 1) / N
        O_h = generate_O_binom(m, N, p_h)
    elif (phi == phi_beta_shifted):            
        p_h = (np.sum(np.diag(unif_partition(n)).dot(C_h), axis = 0) - a) / (1 - 2*a)
        O_h = generate_O_stochastic_N(m, N, p_h)
        
    print 'p_h = ' 
    print p_h   
        
    return O_h
    
def col_normalize(M):

    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))
    
      
def check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123):
    print np.linalg.norm(P_21 - R_21)
    print np.linalg.norm(P_31 - R_31)
    print np.linalg.norm(P_23 - R_23)
    print np.linalg.norm(P_13 - R_13)
    print np.linalg.norm((P_123 - R_123).reshape((1,n*n*n)))
    
    
def estimate(P_21, P_31, P_23, P_13, P_123, m):
    U = range_cons(P_21, m);
    M_2, M_3 = symmetrize(P_21, P_31, P_23, P_13, P_123, U);  
    
    W, X_3 = tentopy.whiten(M_2, M_3); 
    Lambda, U_T_O = tentopy.reconstruct(W, X_3);
    
    O_h = np.dot(U, U_T_O.T) 
    O_h = col_normalize(O_h)
    
    T_h = np.linalg.pinv(O_h).dot(P_21.dot(np.linalg.pinv(O_h.T)))
    T_h = col_normalize(T_h)
    
    return O_h, T_h
    
def estimate_refine(C_h, P_21, phi, N, n, m, a):    
    
    O_h = get_O(phi, N, n, C_h, a)
    C_h_p = gt_obs(phi, N, n, O_h)
    T_h_p = np.linalg.pinv(C_h_p).dot(P_21.dot(np.linalg.pinv(C_h_p.T)))
    T_h_p = col_normalize(T_h_p)
    
    return C_h_p, T_h_p    


def generate_p(m):
    #p = np.asarray([0,0.5,1])
    p = np.asarray([0.3,0.7])
    #print p
    
    return p
    
    
def generate_O_binom(m, N, p):
    O = np.zeros((N+1, m));
    
    for i in xrange(N+1):
        for j in xrange(m):
            O[i,j] = stats.binom.pmf(N, i, p[j])
            #O[i,j] = special.binom(N, i) * (p[j] ** i) * ((1-p[j]) ** (N-i))
       
    return O
    
    
def generate_O_stochastic_N(m, N, p):
    O = np.zeros(((N+1)*(N+1), m))
    
    # n = i possibly take value 0,...,N
    # x = j possibly take value 0,...,N (0, .., i)
    
    for i in xrange(N+1):
        for k in xrange(i+1):
            for j in xrange(m):  
                #v = special.binom(i, k) * (p[j] ** k) * ((1-p[j]) ** (i-k)) / (N+1)
                #if np.isnan(v):
                #    print i,k,j,p[j]
                O[(N+1)*i + k, j] = stats.binom.pmf(k, i, p[j]) / (N+1)
                
                
    return O
    


def generate_O(m, N, min_sigma_o):

    O = dataGenerator.makeObservationMatrix(m, N+1, min_sigma_o)
    #O = np.eye(3);
    #O = np.asarray([[0.5, 0], [0, 0.5], [0.5, 0.5]])
    return O
    
    
def generate_T(m, min_sigma_t):
    #T = dataGenerator.makeTransitionMatrix(m, min_sigma_t)
    #T = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]);
    #T = np.eye(3);
    T = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    
    return T
    
def generate_initDist(m):
    #initDist = dataGenerator.makeDistribution(m)
    #initDist = np.asarray([0.33,0.33,0.34])
    initDist = np.asarray([0.6, 0.4])
    
    return initDist

if __name__ == '__main__':


    np.random.seed(0);
    #N = 3
    m = 5
    n = 5;
    
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
    
    
    print 'Reading Data..'
    filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chrY_binsize100.mat'
    N, X_importance_weighted, a = data_import.data_prep(filename);
    print 'N = '
    print N
    print 'a = '
    print a
    #X = X[:10000,:]
    
    #phi = phi_onehot;
    #phi = phi_beta;
    phi = phi_beta_shifted;
    
    if phi == phi_onehot:
        n = N + 1

    #C = gt_obs(phi, N, n, O);

    print 'Constructing Moments..'    
    P_21, P_31, P_23, P_13, P_123 = moments_cons_importance_weighted(X_importance_weighted, phi, N, n);
    #P_21, P_31, P_23, P_13, P_123, C, S_1, S_3 = moments_gt(O, phi, N, n, T, initDist)
    #R_21, R_31, R_23, R_13, R_123, C, S_1, S_3 = moments_gt(O, phi, N, n, T, initDist)
    
    #print 'C = '
    #print C
    
    #check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123)
    
    print 'Estimating..'
    C_h, T_h = estimate(P_21, P_31, P_23, P_13, P_123, m)
    #C_h, T_h = estimate(R_21, R_31, R_23, R_13, R_123, m)
    print 'C_h = '
    print C_h
    
    print 'T_h = '
    print T_h
    
    print 'Refining using Binomial Knowledge'

    C_h_p, T_h_p = estimate_refine(C_h, P_21, phi, N, n, m, a)
    print 'C_h_p = '
    print C_h_p
    print 'T_h_p = '
    print T_h_p
    #print get_p(phi, N, n, O_h)
   
    
