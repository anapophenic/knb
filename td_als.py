import numpy as np
import numerical_la as nla
from sktensor import dtensor, cp_als
import matplotlib.pyplot as plt
import scipy.io as io
import itertools
import cvxpy as cvx
import visualize as vis
import postprocess as pp
import td_tpm
import utils as ut


def khatri_rao(As):
    k = np.shape(As[0])[1]
    total_dims = ut.prod([np.shape(A)[0] for A in As])
    B = np.zeros((total_dims, k))

    for i in range(k):
        b = np.ones(1);
        for A in As:
            b = np.kron(b, A[:,i])
        B[:,i] = b

    return B

def rand_init(r, dims, k):
    As = []
    for i in range(r):
        As.append(np.random.randn(dims[i], k))
        As[i] = ut.normalize_m_l2(As[i])

    return As

def squeeze(T, dims ,i):
    T_perm = np.moveaxis(T, i, -1)
    T_squeezed = np.reshape(T_perm, (-1, dims[i]))
    return T_squeezed

def regularized_iter(T, As, A_ref, lam, i):
    dims = np.shape(T)
    k = np.shape(As[0])[1]
    As_p = As[:]
    As_p.pop(i)
    B = khatri_rao(As_p)
    T_squeezed = squeeze(T, dims, i)
    As[i] = np.linalg.solve(B.T.dot(B) + lam * np.eye(k), B.T.dot(T_squeezed) + lam * A_ref.T).T
    As[i] = ut.normalize_m_l2(As[i])
    return As

def regularized_iter_noref(T, As, lam, i):
    A_ref = np.zeros(np.shape(As[i]));
    return regularized_iter(T, As, A_ref, lam, i)

def get_w(T, As):
    B_all = khatri_rao(As)
    T_flattened = np.ndarray.flatten(T)
    w = np.linalg.pinv(B_all).dot(T_flattened)
    return w

def residual(T, As):
    B_all = khatri_rao(As)
    T_flattened = np.ndarray.flatten(T)
    w = get_w(T, As)
    return np.linalg.norm(B_all.dot(w) - T_flattened)

def rel_err(T, As):
    return pow(residual(T, As) / np.linalg.norm(T), 2.0)

def triple_eye(k):
  T = np.zeros((k,k,k))

  for j in xrange(k):
    T[j,j,j] = 1

  return T

def triple_diag(w):
    k = np.shape(w)[0]
    T = np.zeros((k,k,k))
    for j in xrange(k):
        T[j,j,j] = w[j]

    return T

def cp_to_tensor(w, As):
    return nla.fast_trilinear(triple_diag(w), As[0].T, As[1].T, As[2].T)

def error_eval_tensor(w_1, As_1, T_1, w_2, As_2, T_2):
    T_1_h = cp_to_tensor(w_1, As_1)
    T_2_h = cp_to_tensor(w_2, As_2)
    return np.linalg.norm(T_1 - T_1_h) + np.linalg.norm(T_2 - T_2_h)

def data_gen(N, k, alpha, r):
    w = np.ones(k)
    Cs_1 = []
    Cs_2 = []
    for i in range(r):
        Cs_1.append(ut.normalize_m_l2(np.random.randn(N,k)))
        Cs_2.append(Cs_1[i] + alpha * ut.normalize_m_l2(np.random.randn(N,k)))

    T_1 = cp_to_tensor(w, Cs_1)
    T_2 = cp_to_tensor(w, Cs_2)

    return T_1, T_2

def data_gen_real(filename):
    P_21, P_31, P_23, P_13, P_123 = vis.load_moments(filename)

    P_21_1 = P_21[:10, :10]
    P_21_2 = P_21[10:20, 10:20]

    P_31_1 = P_31[:10, :10]
    P_31_2 = P_31[10:20, 10:20]

    P_23_1 = P_23[:10, :10]
    P_23_2 = P_23[10:20, 10:20]

    P_13_1 = P_13[:10, :10]
    P_13_2 = P_13[10:20, 10:20]

    P_123_1 = P_123[:10, :10, :10]
    P_123_2 = P_123[10:20, 10:20, 10:20]

    return P_21_1, P_31_1, P_23_1, P_13_1, P_123_1, P_21_2, P_31_2, P_23_2, P_13_2, P_123_2

def plot_error(ls, errs, k, lam, type):
    fig, ax = plt.subplots(1, 1)
    ax.semilogx(ls, errs)
    ax.grid(True)
    ax.set_title('Reconstruction Error, k = ' + str(k) + '_lambda=' + str(lam))

    plt.show(block=False)

    if type == 'synthetic':
        fig.savefig('Synthetic_k=' + str(k) + '_i='+ str(i))
    else:
        fig.savefig('Real_k=' + str(k) + '_i='+ str(i))

    plt.close('all')

def als(P_123, m_hat):
    w, As = als_main(P_123, m_hat, 0, 1e-3)
    return As[1]


'''
If we normalize lazily (in the outer loop), it often gives numerically unstable solutions.
'''
def als_main(T, k, lam, tol, max_iter=1000):
    dims = np.shape(T)
    r = len(dims)
    lam = lam + 1e-3 * np.ones(r)
    As = rand_init(r, dims, k)

    for it in range(max_iter):
        for i in range(r):
            #deep copy
            As = regularized_iter_noref(T, As, lam[i], i)

        if rel_err(T, As) < tol:
            break

    w = get_w(T, As)

    return w, As
'''
Let us first assume T_1 and T_2 are of the same dimension
'''
def co_regularized_als(T_1, T_2, k, lam, tol, max_iter=1000):
    dims = np.shape(T_1)
    r = len(dims)
    lam = lam + 1e-3 * np.ones(r)

    As_1 = rand_init(r, dims, k)
    As_2 = rand_init(r, dims, k)

    for it in range(max_iter):

        for i in range(r):
            As_1 = regularized_iter(T_1, As_1, As_2[i], lam[i], i)
            As_2 = regularized_iter(T_2, As_2, As_1[i], lam[i], i)

        if rel_err(T_1, As_1) < tol and rel_err(T_2, As_2) < tol:
            break

    w_1 = get_w(T_1, As_1)
    w_2 = get_w(T_2, As_2)

    return w_1, As_1, w_2, As_2

if __name__ == '__main__':
    '''
    Parameter Setting
    '''
    tol = 0.0001
    r = 3
    '''
    Instance Generation
    '''
    #T_1, T_2 = data_gen(N, k, alpha, r)

    filename = 'input_tensor.mat'
    P_21_1, P_31_1, P_23_1, P_13_1, P_123_1, P_21_2, P_31_2, P_23_2, P_13_2, P_123_2 = data_gen_real(filename)
    N = np.shape(P_123_1)[0]

    ls = [pow(2,z) for z in range(-15, -14)]
    ks = range(1,8)
    errs = []
    lims = [0, N]

    for k, l in itertools.product(ks, ls):

        # co-regularze on dimension 2
        lam = np.zeros(r)
        lam[1] = l;
        w_1, As_1, w_2, As_2 = co_regularized_als(P_123_1, P_123_2, k, lam, tol)
        #err_1 = ut. normalized_km(As_1[1], Cs_1[1])
        #err_2 = ut.normalized_km(As_2[1], Cs_2[1])
        err_3 = error_eval_tensor(w_1, As_1, P_123_1, w_2, As_2, P_123_2)
        errs.append(err_3)

        color_scheme = vis.default_color_scheme(k)
        As_1[1] = pp.postprocess_m(As_1[1])

        vis.print_feature_map(As_1[1], color_scheme, 'co_regularize/', 'k =' + str(k) + 'lambda = ' + str(l) + '_feature_map.pdf', lims)

        O_avg, H_21 = pp.refine_als_p21(P_21_1, As_1[1])

        print H_21
        print O_avg

        vis.print_feature_map(O_avg, color_scheme, 'co_regularize/', 'k =' + str(k) + 'lambda = ' + str(l) + '_feature_map_refined.pdf', lims)

        O_tpm = td_tpm.tpm(P_21_1, P_31_1, P_23_1, P_13_1, P_123_1, k)
        O_tpm = pp.postprocess_m(O_tpm)

        vis.print_feature_map(O_tpm, color_scheme, 'co_regularize/', 'k =' + str(k) + 'lambda = ' + str(l) + '_feature_map_tpm.pdf', lims)

    plot_error(ls, errs, k)

    '''
    T_h = nla.fast_trilinear(core, As[0].T, As[1].T, As[2].T)
    print 'T - T_h', np.linalg.norm(T - T_h)
    print 'T', np.linalg.norm(T)
    '''

    #w, As = als(T, k, np.zeros(3), tol)

    #for i in range(3):
    #print As[i]

    '''
    print 'Cs = ', Cs
    print 'As = ', As
    '''

    '''
    T_d = dtensor(T)

    # Decompose tensor using CP-ALS
    P, fit, itr, exectimes = cp_als(T_d, 3, init='random')
    print P.U
    print fit
    print itr
    print exectimes

    T_h = P.toarray()
    print 'T - T_h', np.linalg.norm(T - T_h)
    print 'T', np.linalg.norm(T)
    '''

    '''
    T = np.zeros((N,N,N))
    for j in range(k):
    T[j,j,j] = 1
    '''
