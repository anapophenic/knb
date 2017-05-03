import numpy as np
import tentopy
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
    P_123 = nla.trilinear('I', C_1.T, np.dot(C_2, np.diag(np.dot(T, initDist))).T, C_3.T);
    '''

    C = fm.gt_obs(phi, N, n, O);

    R_21 = C.dot(T.dot(np.diag(initDist).dot(C.T)))
    R_31 = C.dot(T.dot(T.dot(np.diag(initDist).dot(C.T))))
    R_32 = C.dot(T.dot(np.diag(T.dot(initDist)).dot(C.T)))
    R_23 = R_32.T
    R_13 = R_31.T

    C_1 = C.dot(np.diag(initDist).dot(T.T.dot(np.linalg.inv(np.diag(T.dot(initDist))))))
    C_2 = C
    C_3 = C.dot(T)

    #R_123 = nla.trilinear('I', C_1.T, C_2.dot(np.diag(T.dot(initDist))).T, C_3.T);
    R_123 = nla.fast_trilinear('I', C_1.T, C_2.dot(np.diag(T.dot(initDist))).T, C_3.T);

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

    #N_3 = timing_val(nla.trilinear) (P_123, U.dot(S_1.T), U, U.dot(S_3.T))
    #M_3 = timing_val(nla.fast_trilinear) (P_123, U.dot(S_1.T), U, U.dot(S_3.T))
    #print abs(M_3 - N_3) < 0.001

    M_3 = nla.fast_trilinear(P_123, U.dot(S_1.T), U, U.dot(S_3.T))

    return M_2, M_3

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
    print 'raw O_h = '
    print O_h


    T_h = np.linalg.pinv(O_h).dot(P_21.dot(np.linalg.pinv(O_h.T)))
    print 'raw T_h = '
    print T_h

    pi_h = np.sum(T_h, axis = 0);
    print 'raw pi_h = '
    print pi_h

    pi_h = pp.postprocess_v(pi_h)
    T_h = pp.postprocess_m(T_h)

    vis.show_T(T_h, 'T_'+str(m), 'merge_ctxts/')
    vis.show_pi(pi_h, 'pi_'+str(m), 'merge_ctxts/')


    T_h, pi_h = pp.recover_T_pi(P_21, O_h)

    vis.show_T(T_h, 'T_p_'+str(m), 'merge_ctxts/')
    vis.show_pi(pi_h, 'pi_p_'+str(m), 'merge_ctxts/')
    

    print T_h
    print pi_h

    O_h = pp.postprocess_m(O_h)

    return O_h, T_h, pi_h

def estimate_refine(C_h, P_21, phi, N, n, m, a):

    O_h = fm.get_O(phi, N, n, C_h, a)
    C_h_p = fm.gt_obs(phi, N, n, O_h)
    T_h_p = np.linalg.pinv(C_h_p).dot(P_21.dot(np.linalg.pinv(C_h_p.T)))
    T_h_p = pp.postprocess_m(T_h_p)

    return C_h_p, T_h_p
