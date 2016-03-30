import dataGenerator
import numpy as np
from scipy import special
import tentopy
import kernelNaiveBayes

def moments_cons(X, phi, N, n):
    
    P_21 = np.zeros((n,n));
    P_31 = np.zeros((n,n));    
    P_23 = np.zeros((n,n));    
    P_13 = np.zeros((n,n));        
    P_123 = np.zeros((n,n,n));         
    
    s = float(np.shape(X)[0]);
    
    for i in range(np.shape(X)[0]):
        p1 = phi(X[i,0], N, n);
        p2 = phi(X[i,1], N, n);
        p3 = phi(X[i,2], N, n);
    
        P_21 += np.kron(p2, p1). reshape(n, n) / s
        P_31 += np.kron(p3, p1). reshape(n, n) / s
        P_23 += np.kron(p2, p3). reshape(n, n) / s
        P_13 += np.kron(p1, p3). reshape(n, n) / s
        P_123 += np.kron(p1, np.kron(p2, p3)). reshape(n, n, n) / s
        
    return np.mat(P_21), np.mat(P_31), np.mat(P_23), np.mat(P_13), P_123

def range_cons(P, m):

    #print P

    U, S, V = np.linalg.svd(P, full_matrices=False)
    return U[:,:m]
    

def symmetrize(P_21, P_31, P_23, P_13, P_123, U):
    Q_21 = U.T * P_21 * U;
    Q_31 = U.T * P_31 * U;
    Q_23 = U.T * P_23 * U;
    Q_13 = U.T * P_13 * U;
    S_1 = Q_23 * np.linalg.inv(Q_21);
    S_3 = Q_21 * np.linalg.inv(Q_23);
    
    M_2 = S_1 * Q_21.T;
    M_2 = (M_2 + M_2.T)/2;
    
    #print np.shape(P_123)
    #Q_123 = np.tensordot(P_123, U, axes=([0], [0]));
    #print np.shape(Q_123)
    #Q_123 = np.tensordot(Q_123, U, axes=([2], [0]));  
    #M_3 = np.tensordot(Q_123, S_1, axes=([0], [1]));
    #M_3 = np.tensordot(M_3, S_3, axes=([2], [1]));
    
    M_3 = kernelNaiveBayes.trilinear(P_123, U*S_1.T, U, U*S_3.T)

    return M_2, M_3
    
def phi_beta(x, N, n):
    #print x
    #print N
    p = np.asarray(map(lambda t: pow(t, x) * pow(1-t, N-x), np.linspace(0.05,0.95,n).tolist()));
    return p / sum(p);

if __name__ == '__main__':

    N = 100
    m = 10
    l = 10000
    min_sigma = 0.1
    T = dataGenerator.makeTransitionMatrix(m, min_sigma)
    initDist = dataGenerator.makeDistribution(m)
    p = np.random.rand(m)
    X = dataGenerator.generateData(N, m, T, p, initDist, l)
    

    P_21, P_31, P_23, P_13, P_123 = moments_cons(X, phi_beta, N, 20);
    U = range_cons(P_21, m);
    M_2, M_3 = symmetrize(P_21, P_31, P_23, P_13, P_123, U);
    
    print np.shape(M_2)
    print np.shape(M_3)
    
    W, X_3 = tentopy.whiten(M_2, M_3);
    
    print np.shape(W)
    print np.shape(X_3)
    Lambda, U_T_O = tentopy.reconstruct(W, X_3);
    
    #postprocessing
    O = U * U_T_O;
    O = O * np.linalg.inv(np.diag(np.sum(O, axis=0)))
    
    p = ((N+1) * np.sum(np.diag(np.linspace(0,1,n)) * O) - 1) / N
    
    print p
    
    
    T = np.linalg.pinv(O) * P_21 * np.linalg.pinv(O.T)
    T = T * np.linalg.inv(np.diag(np.sum(T, axis=0)))
    
    print T
    
    
