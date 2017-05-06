import numpy as np
import scipy
import utils as ut

def generate_p(m):
    #p = np.asarray([0,0.5,1])
    #p = np.asarray([0.3,0.7])
    p = np.asarray(ut.unif_partition(m));
    #print p
    return p

def generate_p_ch(m, r):
    p_ch = np.zeros((r, m));
    for c in range(r):
        p_ch[c,:] = generate_p(m);

    return p_ch


def generate_p_N(N):
    #p = np.zeros(N+1);
    #p[N] = 1;
    p = np.ones(N+1) / (N+1);
    return p

#def generate_p_N_c(N, r):
#    for c in range(r):

def generate_T(m, min_sigma_t):
    #T = dataGenerator.makeTransitionMatrix(m, min_sigma_t)
    #T = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]);
    #T = np.eye(3);
    #T = np.asarray([[0.8, 0.2], [0.2, 0.8]])

    T = min_sigma_t * np.eye(m) + (1 - min_sigma_t) * np.random.random((m, m))
    return ut.normalize_m(T)

def generate_pi(m):
    #pi = dataGenerator.makeDistribution(m)
    #pi = np.asarray([0.33,0.33,0.34])
    #pi = np.asarray([0.6, 0.4])

    pi = np.ones(m) / m;
    return pi

def generate_O(m, n, min_sigma_o):
    O = min_sigma_o * np.eye(n,m) + (1 - min_sigma_o) * np.random.random((n, m))
    return ut.normalize_m(O)

def draw_categorical(values, probabilities):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random(), bins)]

#def generate_seq_p(p, N, T, pi, l):
#    m = np.shape(T)[0];
#    O = get_O_stochastic_N(m, np.ones(N+1)/(N+1), p);
#    return generate_seq(O, T, pi, l)

#def generate_firstfew_p(p, N, T, pi, l):
#    m = np.shape(T)[0];
#    O = get_O_stochastic_N(m, np.ones(N+1)/(N+1), p);
#    return generate_firstfew_p(O, T, pi, l)

#def generate_longchain_p(p, N, T, pi, l):
#    m = np.shape(T)[0];
#    O = get_O_stochastic_N(m, np.ones(N+1)/(N+1), p);
#    return generate_longchain(O, T, pi, l)


# generate a chain (of hidden states) of length l
def generate_hidden(T, pi, l):

    m = np.shape(pi)[0];
    h = np.int_(np.zeros(l));
    for i in range(l):
        if i == 0:
            h[i] = draw_categorical(range(m), pi);
        else:
            h[i] = draw_categorical(range(m), T[:,h[i-1]].reshape(m));

    return h;

# generate a chain (of hidden states and observations) of length l
def generate_seq(O, T, pi, l):

    n = np.shape(O)[0];
    x = np.int_(np.zeros(l));

    h = generate_hidden(T, pi, l);

    for i in range(l):
        x[i] = draw_categorical(range(n), O[:,h[i]].reshape(n));

    return x, h

# given a distribution on coverage p_N,
# generate a chain of hidden states
def generate_seq_bin_c(p_ch, p_N, T, pi, l):

    h = generate_hidden(T, pi, l);
    r = np.shape(p_ch)[0];
    N = np.shape(p_N)[0] - 1; #Values that coverage can take: [0, 1, ..., N]

    coverage = np.zeros((r, l));
    methylated = np.zeros((r, l));

    for c in range(r):
        for i in range(l):
            coverage[c, i] = draw_categorical(range(N+1), p_N);
            methylated[c, i] = np.random.binomial(coverage[c, i], p_ch[c, h[i]])

    return coverage, methylated, h



def generate_firstfew(O, T, pi, l):
    ### This function generates l triples, each representing methylation counts at first 3 sites of a sequence
    ### Data generated from an HMM with transition matrix T and obervation matrix O
    #   T and O are column stochastic
    x_zipped = [];

    for i in range(l):
        x, h = generate_seq(T, O, pi, 3)
        x_zipped.append((x[0], x[1], x[2]))

    return x_zipped

def generate_longchain(O, T, pi, l):

    x, h = generate_seq(T, O, pi, l+2);
    x_zipped = np.array(zip(x[0:l], x[1:l+1], x[2:l+2]))

    return x_zipped
