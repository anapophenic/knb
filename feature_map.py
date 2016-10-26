import numpy as np

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
    #print x
    #print N
    i = int(x / (N+1));
    k = int(x) % (N+1);
    if k > i:
        return np.zeros(n)

    #p = np.asarray(map(lambda t: (t ** k) * ( (1-t) ** (i - k) ), unif_partition(n).tolist()));
    p = np.asarray(map(lambda t: beta_interval(t, k, i-k, n), unif_partition(n).tolist()));

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
    #print x
    #print N

    #p = np.asarray(map(lambda t: (t ** x) * ( (1-t) ** (N-x) ), unif_partition(n).tolist()));
    p = np.asarray(map(lambda t: beta_interval(t, x, N-x, n), unif_partition(n).tolist()));
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

def gt_obs(phi, N, n, O):
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
        Trans = np.zeros((n, N+1));
        for x in xrange(N+1):
            Trans[:,x] = phi(x, N, n).T
    elif phi == phi_beta_shifted or phi == phi_binning or phi == phi_beta_shifted_cached or phi == phi_binning_cached:
        Trans = np.zeros((n, (N+1)*(N+1)));
        for x in xrange((N+1)*(N+1)):
            Trans[:,x] = phi(x, N, n).T

    #print 'Trans = '
    #print Trans
    #print 'O = '
    #print O

    O_m = Trans.dot(O);
    return O_m;

def get_a(N):
    return sum(map(lambda n: 1.0/(n+2), range(0,N+1,1))) / (N+1)

def beta_interval(t, k, l, n):
    return stats.beta.cdf(t+0.5/n, k+1, l+1) - stats.beta.cdf(t-0.5/n, k+1, l+1)
