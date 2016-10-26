import numpy as np

def unif_partition(n):
    return np.linspace(1.0/(2*n), 1.0 - 1.0/(2*n), n)

def col_normalize(M):
    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))

#Need some sort of smoothing on the estimated probabilities
def col_normalize_post(M):
    # first, determine the sign of the observations

    s = np.diag(np.sign(np.sum(M, axis=0)))
    M = M.dot(s)

    # second, zero out those negative entries
    M = M * (M > 0)

    # then normalize all the entries
    return col_normalize(M);

def normalize_post(p):
    s = np.sign(np.sum(p))
    p = p * s;
    p = p * (p > 0)

    return p / np.sum(p)

def proj_zeroone(p):
    m = np.shape(p)[0];
    for i in range(m):
        if p[i] > 1:
            p[i] = 1
        elif p[i] < 0:
            p[i] = 0

    return p

def generate_O_binom(m, N, p):
    O = np.zeros((N+1, m));

    for i in xrange(N+1):
        for j in xrange(m):
            O[i,j] = stats.binom.pmf(N, i, p[j])
            #O[i,j] = special.binom(N, i) * (p[j] ** i) * ((1-p[j]) ** (N-i))

    return O


def generate_O_stochastic_N(m, p_N, p):
    N = np.shape(p_N)[0] - 1
    O = np.zeros(((N+1)*(N+1), m))

    # n = i possibly take value 0,...,N
    # x = j possibly take value 0,...,N (0, .., i)

    for i in xrange(N+1):
        for k in xrange(i+1):
            for j in xrange(m):
                #v = special.binom(i, k) * (p[j] ** k) * ((1-p[j]) ** (i-k)) / (N+1)
                #if np.isnan(v):
                #    print i,k,j,p[j]
                O[(N+1)*i + k, j] = stats.binom.pmf(k, i, p[j]) * p_N[i]


    return O



def generate_O(m, N, min_sigma_o):

    O = dataGenerator.makeObservationMatrix(m, N+1, min_sigma_o)
    #O = np.eye(3);
    #O = np.asarray([[0.5, 0], [0, 0.5], [0.5, 0.5]])
    return O
