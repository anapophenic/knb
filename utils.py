import numpy as np

def normalize_m(M):
    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))

def normalize_v(p):
    return p / float(np.sum(p))

def unif_partition(n):
    return np.linspace(0.5/n, 1.0 - 0.5/n, n)
