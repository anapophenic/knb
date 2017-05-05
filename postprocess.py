import cvxpy as cvx
import numpy as np

#Need some sort of smoothing on the estimated probabilities
def postprocess_m(M):
    # first, determine the sign of the observations

    s = np.diag(np.sign(np.sum(M, axis=0)))
    M = M.dot(s)

    # second, zero out those negative entries
    M = M * (M > 0)

    # then normalize all the entries
    M = normalize_m(M);
    M = make_positive_m(M);
    M = normalize_m(M);

    return M

def postprocess_v(p):
    s = np.sign(np.sum(p))
    p = p * s;
    p = p * (p > 0)

    #normalize
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


#make the entries bounded away from zero
def make_positive_v(p):
    m = np.shape(p)[0];
    p = p + 0.0001 * np.ones(m)
    return p

def make_positive_m(M):
    n, m = np.shape(M);
    M = M + 0.0001 * np.ones((n, m))
    return M


def normalize_m(M):
    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))

def normalize_v(p):
    return p / float(np.sum(p))

def normalize_m_all(p):
    return p / float(np.sum(p))


def recover_T_pi(P_21, O_h):
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

if __name__ == '__main__':
    C_h = np.eye(4,2)
    P_21 = np.eye(4,4)
    print C_h
    print P_21
    recover_T_pi(P_21, C_h)
