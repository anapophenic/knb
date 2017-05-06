import numpy as np
import numerical_la as nla
import feature_map as fm
import binom_hmm as bh
import postprocess as pp
import visualize as vis

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
        if i % 10000 == 0:
            print str(i) + ' samples have been added to moment matrix.'
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

    '''
    Formulae:
    P_21 = C * T * diag(initDist) * C.T
    P_31 = C * T * T * diag(initDist) * C.T

    P_32 = C * T * diag(T * initDist) * C.T
    P_23 = P_32.T
    P_13 = P_31.T

    C_1 = C * diag(initDist) * T.T * diag(T*initDist)^{-1}
    C_2 = C
    C_3 = C * T

    P_123 = I (C_1, C_2 * diag(T * initDist), C_3)
    '''

def moments_gt(O, phi, N, n, T, initDist):


    C = fm.gt_obs(phi, N, n, O);

    R_21 = C.dot(T.dot(np.diag(initDist).dot(C.T)))
    R_31 = C.dot(T.dot(T.dot(np.diag(initDist).dot(C.T))))
    R_32 = C.dot(T.dot(np.diag(T.dot(initDist)).dot(C.T)))
    R_23 = R_32.T
    R_13 = R_31.T

    C_1 = C.dot(np.diag(initDist).dot(T.T.dot(np.linalg.inv(np.diag(T.dot(initDist))))))
    C_2 = C
    C_3 = C.dot(T)

    R_123 = nla.fast_trilinear('I', C_1.T, C_2.dot(np.diag(T.dot(initDist))).T, C_3.T);

    S_1 = C_2.dot(np.linalg.pinv(C_1))
    S_3 = C_2.dot(np.linalg.pinv(C_3))

    return R_21, R_31, R_23, R_13, R_123, C, S_1, S_3

def check_conc(P_21, R_21, P_31, R_31, P_23, P_13, P_123, R_123):
    print np.linalg.norm(P_21 - R_21)
    print np.linalg.norm(P_31 - R_31)
    print np.linalg.norm(P_23 - R_23)
    print np.linalg.norm(P_13 - R_13)
    print np.linalg.norm((P_123 - R_123).reshape((1,n*n*n)))
