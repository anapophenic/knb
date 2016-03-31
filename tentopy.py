#!/usr/bin/env/python
"""
linalg.py

Linear algebra tools for orthogonal tensor decompositions.

The power method algorithm is based on the robust tensor power method, described
in Algorithm 1 of "Tensor Decompositions for Learning Latent Variable Models"
by Anandkumar et al.
"""

import numpy as np
import math

def reconstruct(W, X3, L=25, N=20):
  """ Reconstruct the eigenvalues and eigenvectors corresponding to the
  probability distributions.

  inputs:
  W: the whitening matrix of M2
  M3: the third-order moment matrix

  outputs:
  eigenvalues
  eigenvectors
  """
  evecs, evals = eig(X3, L, N)
  evals_rec = np.array([(1. / (w ** 2)) for w in evals])
  evecs_rec = [np.linalg.solve(W.T, e * evecs[k, :]) for k, e in enumerate(evals)]
  # now in reverse order
  return evals_rec[::-1], np.array(evecs_rec[::-1])
  

def whiten(M2, M3):
  """ Form the pseudo-whitening matrix of M2 and apply to M3 to form \tilde{M3}.
  To pseudo-whitening matrix is formed by a thresholding eigenvalue
  decomposition.  If M2 = UDU^T, then form [D']_i = max(abs([D]_i), \epsilon).

  inputs:
  M2: the second-order moment matrix
  M3: the third-order moment matrix
  
  outputs:
  W: the pseudo-whitening matrix
  \tilde{M3}: M3(W, W, W)
  """
  evals, evecs = np.linalg.eig(M2)
  wp = np.diag([1 / math.sqrt(max(abs(w), 10e-12)) for w in evals])
  W = np.dot(evecs, wp)

  # now apply W in all directions to M3
  # TODO: use np dot products
  N1 = W.shape[1]
  N2 = M3.shape[0]
  X3 = tensor_outer(np.zeros(N1), 3)

  # TODO: figure out the equivalent numpy routines
  for i1 in xrange(N1):
    for i2 in xrange(N1):
      for i3 in xrange(N1):
        for j1 in xrange(N2):
          for j2 in xrange(N2):
            for j3 in xrange(N2):
              X3[i1, i2, i3] += M3[j1, j2, j3] * W[j1, i1] * W[j2, i2] * W[j3, i3]

  return W, X3

def approx_eval(T, v):
  """ Compute the approximate eigenvalue corresponding to an approximate
  eigenvector.
  
  inputs:
  T: a super symmetric tensor
  v: an approximate eigenvalue
  
  output:
   approximate eigenvalue T(v, v, v)
  """
  return np.dot(np.tensordot(T, np.outer(v, v)), v)

def tensor_outer(v, n):
  """ Compute a rank-1 order-n (n > 1) tensor computation by outer products.
  
  input: v the basis vector
  output: \otimes^3 v
  """
  T = np.outer(v, v)
  for i in xrange(n - 2):
    T = np.outer(v, T)
  return T.reshape([len(v)] * n)

def power_method(T, L, N, norm_type=2):
  """ Main power method driver.  Computes the (eigenvalue, eigenvector) pair
  of a tensor corresponding to the largest eigenvalue.
  
  inputs:
  T: a super-symmetric tensor
  L: number of inner iterations of power method to perform
  N: number of iterations per inner iteration
  
  outputs:
  a tuple:
      (approximate largest eigenvalue, corresponding eigenvector,
       deflated tensor)
  """
  k = T.shape[0]
  n = len(T.shape)
  thetas = []

  def inner_iter(N, theta):
    for t in xrange(N):
      next_iter = np.tensordot(T, np.outer(theta, theta))
      theta = next_iter / np.linalg.norm(next_iter, norm_type)
    return theta.T

  for tau in xrange(L):
    # Choose a starting vector unfiormly at random from unit ball
    v = np.random.randn(k)
    theta_0 = v / np.linalg.norm(v, n)
    theta_0 = theta_0.reshape((k, 1))
    thetas.append(inner_iter(N, theta_0))

  ind = np.argmax([approx_eval(T, theta) for theta in thetas])
  theta_hat = inner_iter(N, thetas[ind])
  lambda_hat = approx_eval(T, theta_hat)
  rank1_approx = lambda_hat * tensor_outer(theta_hat, n)

  return theta_hat, lambda_hat, T - rank1_approx

def eig(T, L=10, N=10, norm_type=2):
  """ Compute the eigen-decomposition of a super-symmetric tensor.

  inputs:
  T: a super-symmetric tensor
  L: number of inner iterations of power method to perform
  N: number of iterations per inner iteration

  outputs:
  a tuple of eigenvectors and eigenvalues
  """
  if sum([d == T.shape[0] for d in T.shape]) != len(T.shape):
    raise Exception('Each tensor dimension must be the same')
  k = T.shape[0]
  evecs = []
  evals = []
  for i in xrange(k):
    evec, eval, def_T = power_method(T if i == 0 else def_T, L, N, norm_type)
    evecs.append(list(evec))
    evals.append(eval)
  return np.array(evecs), np.array(evals)

if __name__ == '__main__':
  N = 20
  T = tensor_outer(np.zeros(N), 3)

  for j in xrange(N):
    T[j][j][j] = j * N + 1
  print eig(T, norm_type=1)
