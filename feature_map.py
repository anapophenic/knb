import numpy as np
import utils as ut
from scipy import stats
import time
import postprocess as pp
import binom_hmm as bh

def cache_results(a_func):
    '''This decorator funcion binds a map between the tuple of arguments
       and results computed by aFunc for those arguments'''
    def cached_func(*args):
        if not hasattr(a_func, '_cache'):
            a_func._cache = {}
        if args in a_func._cache:
            return a_func._cache[args]
        new_val = a_func(*args)
        a_func._cache[args] = new_val
        return new_val
    return cached_func

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print func.__name__ + 'takes' + str(t2 - t1)
        return res
    return wrapper

def phi_beta_shifted_cached(*args):
    return cache_results(phi_beta_shifted)(*args)

def phi_binning_cached(*args):
    return cache_results(phi_binning)(*args)

def phi_binning_igz_cached(*args):
    return cache_results(phi_binning_igz)(*args)

# dealing with (0,0) observation is a bit tricky. Here we create a new dimension for these obs.
def phi_beta_shifted_cached_listify(*args):
    return phi_listify(phi_beta_shifted_cached)(*args)

def phi_binning_cached_listify(*args):
    return phi_listify(phi_binning_cached)(*args)

def phi_binning_igz_cached_listify(*args):
    return phi_listify(phi_binning_igz_cached)(*args)

# automatically extend phi to domains where there is a list
# basically, split the dimensions to all the contexts equally
# This only works if n is divisible by the number of contexts
# Otherwise the actual total dimension of feature map would not be n!
def phi_listify(phi):
    return lambda xs, N, n: np.ndarray.flatten(np.array([phi(x, N, n/len(list(xs))) for x in list(xs)]))

def phi_lims(n, r):
    return range(0, n+1, n/r)

def phi_binning(x, N, n):
    p = np.zeros(n);
    p[:-1] = phi_binning_igz(x, N, n-1)

    k = int(x) % (N+1);
    if k == 0:
        p[-1] = 1

    return p;

'''
    i = int(x / (N+1));
    k = int(x) % (N+1);
    p = np.zeros(n)

    if k > i or k == 0:
	    p[n-1] = 1
    else:
        prob = float(k) / i;
        #print prob
        for j in range(n-1):
	    if (j <= (n-3) and prob >= float(j) / (n-1) and prob < float(j+1) / (n-1)):
	        p[j] = 1
	    if (j == (n-2) and prob >= float(j) / (n-1) and prob <= float(j+1) / (n-1)):
	        p[j] = 1

    return p;
'''

def phi_binning_igz(x, N, n):
    i = int(x / (N+1));
    k = int(x) % (N+1);
    p = np.zeros(n)

    if k > i or k == 0:
        pass
    else:
        prob = float(k) / i;
        #print prob
        for j in range(n-1):
	    if (j <= (n-2) and prob >= float(j) / n and prob < float(j+1) / n):
	        p[j] = 1
	    if (j == (n-1) and prob >= float(j) / n and prob <= float(j+1) / n):
	        p[j] = 1

    return p;


def phi_beta_shifted(x, N, n):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map
        Output:
            beta-distribution encoding of phi(x)
    '''
    i = int(x / (N+1));
    k = int(x) % (N+1);
    if k > i:
        return np.zeros(n)

    p = np.asarray(map(lambda t: beta_interval(t, k, i-k, n), ut.unif_partition(n).tolist()));

    return p / sum(p);


def phi_beta(x, N, n):
    '''
        Beta mapping that works for a fixed coverage value
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map
        Output:
            beta-distribution encoding of phi(x)
    '''
    p = np.asarray(map(lambda t: beta_interval(t, x, N-x, n), ut.unif_partition(n).tolist()));
    return p / sum(p);

def phi_onehot(x, N, n):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map
        Output:
            one hot encoding of phi(x)
    '''
    p = np.zeros(n);
    p[int(x)] = 1
    return p;

def expected_fm_O(phi, N, n, O):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            n: dimensionality of feature map
            O: one hot observation matrix
        Output:
            O_m: feature obseravtion matrix
    '''

    if phi == phi_onehot or phi == phi_beta:
        transfer = np.zeros((n, N+1));
        for x in xrange(N+1):
            transfer[:,x] = phi(x, N, n).T
    elif phi == phi_beta_shifted or phi == phi_binning or phi == phi_beta_shifted_cached or phi == phi_binning_cached:
        transfer = np.zeros((n, (N+1)*(N+1)));
        for x in xrange((N+1)*(N+1)):
            transfer[:,x] = phi(x, N, n).T

    O_m = transfer.dot(O);
    return O_m;

def expected_fm_p_c(phi, n, p_c, p_h):
    O = bh.get_O_stochastic_N(p_c, p_h)
    N = np.shape(p_c)[0] - 1
    O_m = expected_fm_O(phi, N, n, O)
    return O_m

def expected_fm_p_c_group(phi, n, p_c, p_ch):
    r, m = np.shape(p_ch)
    lims = phi_lims(n, r)
    O_m_group = np.zeros(n, m)

    for i in range(r):
        O_m_group[lims[i]:lims[i+1],:] = expected_fm_p_c(phi, n, p_c, p_ch[i,:])

    return O_m_group

def get_a(N):
    return sum(map(lambda n: 1.0/(n+2), range(0,N+1,1))) / (N+1)

def beta_interval(t, k, l, n):
    return stats.beta.cdf(t+0.5/n, k+1, l+1) - stats.beta.cdf(t-0.5/n, k+1, l+1)

def phi_name(phi):
    if phi == phi_onehot:
        return "onehot"
    elif phi == phi_beta:
        return "beta_fixN"
    elif phi == phi_beta_shifted or phi == phi_beta_shifted_cached or phi == phi_beta_shifted_cached_listify:
        return "beta_full"
    elif phi == phi_binning or phi == phi_binning_cached or phi == phi_binning_cached_listify:
        return "binning"


def get_O(phi, p_c, C_h, a):
    '''
        Input:
            phi: feature map
            [0..N]: possible values x can take
            C_h: estimated observation matrix
        Output:
            p_h: estimated methylating probability
    '''
    N = len(p_c) - 1
    p_h = get_p(phi, N, C_h, a);
    O_h = get_O_from_p(phi, p_c, p_h)

    return O_h

def get_O_from_p(phi, p_c, p_h):

    N = len(p_c) - 1
    if (phi == phi_onehot or phi == phi_beta):
        O_h = get_O_binom(N, p_h)
    elif (phi == phi_beta_shifted or phi == phi_binning):
        O_h = get_O_stochastic_N(p_c, p_h)

    return O_h

def get_p(phi, N, C_h, a):

    n, m = np.shape(C_h);

    #Re-normalize the C_h so that the multiple contexts case works correctly
    #In multiple contexts case, we contatenate the feature representations. Therefore,
    #for extracting the parameters, we need to break them to respective contexts.
    #Important step: renormalize the feature maps
    C_h = pp.postprocess_m(C_h);

    if (phi == phi_onehot):
        p_h = np.sum(np.diag(np.linspace(0,N,N+1)).dot(C_h), axis = 0) / N
    elif (phi == phi_beta):
        p_h = ((N+1) * np.sum(np.diag(ut.unif_partition(n)).dot(C_h), axis = 0) - 1) / N
    elif (phi == phi_beta_shifted or phi == phi_beta_shifted_cached or phi == phi_beta_shifted_cached_listify):
        p_h = (np.sum(np.diag(ut.unif_partition(n)).dot(C_h), axis = 0) - a) / (1 - 2*a)
    elif (phi == phi_binning_igz or phi == phi_binning_igz_cached or phi == phi_binning_igz_cached_listify):
        p_h = np.sum(np.diag(ut.unif_partition(n)).dot(C_h), axis = 0)
    elif (phi == phi_binning or phi == phi_binning_cached or phi == phi_binning_cached_listify):
        p_h = np.sum(np.diag(ut.unif_partition(n-1)).dot(C_h[:-1,:]), axis = 0)

    p_h = pp.proj_zeroone(p_h)

    return p_h

def get_pc(phi, N, C_h, a, lims):
    r = len(lims)-1;
    print lims
    print a
    return np.asarray([get_p(phi, N, C_h[lims[c]:lims[c+1],:], a[c]) for c in range(r)])
