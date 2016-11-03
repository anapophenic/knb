import numpy as np
from scipy import stats

def unif_partition(n):
    return np.linspace(1.0/(2*n), 1.0 - 1.0/(2*n), n)

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

def normalize_m(M):
    return M.dot(np.linalg.inv(np.diag(np.sum(np.asarray(M), axis=0))))

def normalize_v(p):
    return p / float(np.sum(p))

def normalize_m_all(p):
    return p / float(np.sum(p))

#make the entries bounded away from zero
def make_positive_v(p):
    m = np.shape(p)[0];
    p = p + 0.01 * np.ones(m)
    return p

def make_positive_m(M):
    n, m = np.shape(M);
    M = M + 0.01 * np.ones((n, m))
    return M

def get_O_binom(m, N, p):
    O = np.zeros((N+1, m));

    for i in xrange(N+1):
        for j in xrange(m):
            O[i,j] = stats.binom.pmf(N, i, p[j])
            #O[i,j] = special.binom(N, i) * (p[j] ** i) * ((1-p[j]) ** (N-i))

    return O


def get_O_stochastic_N(p_N, p):
    m = np.shape(p)[0]
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

def ctxt_name(ctxt):
    s = ""
    for c in ctxt:
        s = s + str(c)

    return s

#binomial hmm function returning a particular row of observation matrix depending
#on the position of the observation
def p_x_h_binom(p_h, coverage, methylated, i):
    #print coverage[i]
    #print methylated[i]
    #print 'p = '
    #print p_h[j]
    #print 'o = '
    #print O_x[j]
    m = np.shape(p_h)[0];
    O_x = np.zeros(m);
    for j in range(m):
        O_x[j] = stats.binom.pmf(methylated[i], coverage[i], p_h[j])
    return O_x

def p_x_ch_binom(p_ch, coverage, methylated, i):
    '''
    p_ch: c * m matrix
    methylated, coverage: l * c matrix
    c: #contexts
    m: #hidden states
    l: length of the sequence
    '''

    r, m = np.shape(p_ch)[1];
    O_x = np.ones(m);
    for j in range(m):
        for c in range(r):
            O_x[j] = O_x[j] * stats.binom.pmf(methylated[i,c], coverage[i,c], p_ch[c,j])

    return O_x

#hmm function returning a particular row of observation matrix depending
#on the position of the observation
def p_x_h_O(O, x, i):
    return O[x[i],:]

def to_x(c, m, N):
    x = c * (N+1) + m
    return x

def to_c_m(x, N):
    c = int(x / (N+1))
    m = int(x) % (N+1)
    return c, m
