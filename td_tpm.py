import numpy as np
import tentopy
import numerical_la as nla


def range_cons(P, m):
    U, S, V = np.linalg.svd(P, full_matrices=False)
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

    M_3 = nla.fast_trilinear(P_123, U.dot(S_1.T), U, U.dot(S_3.T))

    return M_2, M_3


def tpm(P_21, P_31, P_23, P_13, P_123, m):
    U = range_cons(P_21, m);
    M_2, M_3 = symmetrize(P_21, P_31, P_23, P_13, P_123, U);
    W, X_3 = tentopy.whiten(M_2, M_3);
    Lambda, U_T_O = tentopy.reconstruct(W, X_3);
    O_h = np.dot(U, U_T_O.T)
    return O_h
