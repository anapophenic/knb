import numpy as np
import operator
from scipy.optimize import linear_sum_assignment

def normalize_m(M):
    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))

def normalize_m_l2(A):
    d, r = np.shape(A)
    for i in range(r):
        A[:,i] = A[:,i] / np.linalg.norm(A[:,i])

    return A

def normalize_v(p):
    return p / float(np.sum(p))

def unif_partition(n):
    return np.linspace(0.5/n, 1.0 - 0.5/n, n)

def prod(arr):
    return reduce(operator.mul, arr, 1)

def sign_dist(a, b):
    return min(np.linalg.norm(a-b), np.linalg.norm(a+b))

def normalized_km(A, B):
    A_n = normalize_m_l2(A)
    B_n = normalize_m_l2(B)
    return km(A_n, B_n)

def km(A, B):
    k = np.shape(A)[1]
    dist = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            dist[i,j] = sign_dist(A[:,i], B[:,j])

    row_ind, col_ind = linear_sum_assignment(dist)
    errs = dist[row_ind, col_ind].sum()

    return col_ind, errs

def find_match(A, B):
    col_ind, errs = km(A, B)
    return col_ind

def error_eval(A, B):
    col_ind, errs = km(A, B)
    return errs
