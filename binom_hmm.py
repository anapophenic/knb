import numpy as np
from scipy import stats

def unif_partition(n):
    return np.linspace(0.5/n, 1.0 - 0.5/n, n)

def get_O_binom(m, N, p):
    O = np.zeros((N+1, m));

    for i in xrange(N+1):
        for j in xrange(m):
            O[i,j] = stats.binom.pmf(i, N, p[j])
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

def ctxt_name(ctxt_group):
    s = ''
    for ctxt in ctxt_group:
        s = s + '|'
        for c in ctxt:
            s = s + str(c)

    print s
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
    p_ch: r * m matrix
    methylated, coverage: r * l matrix
    r: #contexts
    m: #hidden states
    l: length of the sequence
    '''

    r, m = np.shape(p_ch);
    O_x = np.ones(m);
    for j in range(m):
        for c in range(r):
            #print '----------------'
            #print methylated[c,i], coverage[c,i], p_ch[c,j]
            O_x[j] = O_x[j] * stats.binom.pmf(methylated[c,i], coverage[c,i], p_ch[c,j])

    # Sometimes it will return O with row zero. For robustness, we output [1,1,..,1].
    if np.sum(O_x) == 0:
        O_x = np.ones(m);

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
