import cvxpy as cvx
import numpy as np
import visualize as vis
from utils import normalize_m, normalize_v

#Smoothing on the estimated probabilities
def postprocess_m(M):
    # first, determine the sign of the observations
    # second, zero out those negative entries
    # then normalize all the entries

    s = np.diag(np.sign(np.sum(M, axis=0)))
    M = M.dot(s)

    M = M * (M > 0)

    M = normalize_m(M);
    M = make_positive_m(M);
    M = normalize_m(M);

    return M

def postprocess_v(p):
    s = np.sign(np.sum(p))
    p = p * s;
    p = p * (p > 0)

    p = normalize_v(p);
    p = make_positive_v(p);
    p = normalize_v(p);

    return p

def proj_zeroone(p):
    m = np.shape(p)[0];
    for i in range(m):
        if p[i] > 0.99:
            p[i] = 0.99
        elif p[i] < 0.01:
            p[i] = 0.01

    return p

def make_positive_v(p):
    m = np.shape(p)[0];
    p = p + 0.0001 * np.ones(m)
    return p

def make_positive_m(M):
    n, m = np.shape(M);
    M = M + 0.0001 * np.ones((n, m))
    return M


def normalize_m_all(p):
    return p / float(np.sum(p))

def row_col_normalize_l1(A):
    return A / np.sum(A)


def refine_positify(O_h, P_21, P_31, P_23, P_13, P_123, m):

    print 'raw O_h = '
    print O_h

    T_h = np.linalg.pinv(O_h).dot(P_21.dot(np.linalg.pinv(O_h.T)))
    print 'raw T_h = '
    print T_h

    pi_h = np.sum(T_h, axis = 0);
    print 'raw pi_h = '
    print pi_h

    pi_h = postprocess_v(pi_h)
    T_h = postprocess_m(T_h)

    vis.show_T(T_h, 'T_'+str(m), 'merge_ctxts/')
    vis.show_pi(pi_h, 'pi_'+str(m), 'merge_ctxts/')


    #T_h, pi_h = refine_nmf(P_21, O_h)
    #vis.show_T(T_h, 'T_p_'+str(m), 'merge_ctxts/')
    #vis.show_pi(pi_h, 'pi_p_'+str(m), 'merge_ctxts/')


    print T_h
    print pi_h

    O_h = postprocess_m(O_h)

    return O_h, T_h, pi_h

def refine_prob_model(C_h, P_21, phi, N, n, m, a):

    O_h = fm.get_O(phi, N, n, C_h, a)
    C_h_p = fm.gt_obs(phi, N, n, O_h)
    T_h_p = np.linalg.pinv(C_h_p).dot(P_21.dot(np.linalg.pinv(C_h_p.T)))
    T_h_p = postprocess_m(T_h_p)

    return C_h_p, T_h_p


def refine_nmf(P_21, O_h):
    n, m = np.shape(O_h)
    H_21 = cvx.Variable(m,m)
    constraints = [H_21 >= 0, cvx.sum_entries(H_21) == 1]
    objective = cvx.Minimize( cvx.norm(O_h * H_21 * O_h.T - P_21, 'fro'))
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()

    H_21 = np.asarray(H_21.value)
    pi = np.sum(H_21, axis=0)
    T = normalize_m(H_21)

    return T, pi

def refine_als_p21(P_21, O_init, iters=30):
    n = np.shape(P_21)[0]
    m = np.shape(O_init)[1]

    O_1 = O_init
    O_2 = O_init
    H_21 = row_col_normalize_l1(np.random.rand(m,m))
    residual = np.zeros(iters)

    for it in range(iters):
        if it % 3 == 1:
            O_1 = cvx.Variable(n, m)
            constraint = [O_1 >= 0, cvx.sum_entries(O_1, axis=0) == 1]
        elif it % 3 == 2:
            O_2 = cvx.Variable(n, m)
            constraint = [O_2 >= 0, cvx.sum_entries(O_2, axis=0) == 1]
        else:
            H_21 = cvx.Variable(m, m)
            constraint = [H_21 >= 0, cvx.sum_entries(H_21) == 1]
            O_avg = (O_1 + O_2) / 2
            O_1 = O_avg
            O_2 = O_avg


        obj = cvx.Minimize(cvx.norm(P_21 - O_1 * H_21 * O_2.T))
        prob = cvx.Problem(obj, constraint)
        prob.solve()

        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")

        print 'Iteration {}, residual norm {}'.format(it, prob.value)
        residual[it] = prob.value

        if it % 3 == 1:
            O_1 = O_1.value
        elif it % 3 == 2:
            O_2 = O_2.value
        else:
            H_21 = H_21.value

    return O_avg, H_21

if __name__ == '__main__':
    C_h = np.eye(4,2)
    P_21 = np.eye(4,4)
    print C_h
    print P_21
    recover_T_pi(P_21, C_h)
