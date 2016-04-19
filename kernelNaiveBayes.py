###############################################################################
# Implementation of kernel multi-view spectral algorithm of Song et al. (2014)
#
# Author: E.D. Gutierrez (edg@icsi.berkeley.edu)
# Created: 24 March 2016
# Last modified: 29 March 2016
#
# Sample usage: see knbTest.py.  The main functions are kernHMM and kernXMM
# 
###############################################################################
import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.spatial.distance import pdist,cdist,squareform
import sys; sys.path.append('c:/users/e4gutier/documents/')
import tentopy
import itertools


inner, outer = 100,100  #number of inner and outer iterations for tensor power method

pdist2 = lambda X,Y: squareform(pdist(X,Y)) #compute square form of pairwise distances

# <codecell>

def kernHMM(X,k,kernel='gaussian',symmetric=False,var=1, xRange=None):
  """
  Main function for learning Hidden Markov Model with D-dimensional obs.
  Inputs:
     X: D x (m+2) matrix with m+2 samples of dimension D 
     k: rank of model (number of hidden states)
     symmetric: whether the model has symmetric views or not (should be False
         for HMM)
     xRange: range of X's for which to compute p(x|h). each row is a point.
     var: variance of the kernel (for most kernels).  for beta kernel, var is a
         list with var[0] = numObs and var[1] = desired dimensionality of 
         output feature map.
  Otputs:
     pXbarH:  probabilitiy densities of hidden states (i.e., p(x|h)) at points
         specified by xrange. This is len(xrange) x k array.
  """    
  xRange = np.matrix(xRange)
  if xRange.shape[0]==1:
      xRange = xRange.T
  (K,L,G) = computeKerns(X,kernel,symmetric,var)
  (A,pi) = kernSpecAsymm(K,L,G,k)
  if xrange is None:
    pXbarH = G*A
  else:
    pXbarH = crossComputeKerns(xRange,np.matrix(X)[2:,:])*A
  return pXbarH  

# <codecell>

def kernXMM(X,k,queryX=None, kernel='gaussian', var=1, symmetric=False):
  """
  Main function for learning three-view mixture model 
  Inputs:
     X: D x 3 matrix with m observatioons from each of 3 views
     k: rank of model (number of hidden states)
  Optional inputs:
     queryX: D_test x 1 matrix of test points for which to comptue p(x|h). If 
             queryX is None, queryX is set to X[:,2]
     kernel: kernel to use for smoothing probability distributions p(x|h)
     var: variance of kernel (smoothing).
     symmetric: whether to use the symmetric version of algorithm (set to False)
  Otputs:
     pXbarH: probabilitiy densities of hidden states
  """    
  (K,L,G) = computeKerns3(X,kernel,symmetric,var)
  if symmetric:
    (A,pi) = kernSpecSymm(np.hstack(K,L), np.hstack(L,K),k)
  else:
    (A,pi) = kernSpecAsymm(K,L,G,k)
  if queryX is None:
    pXbarH = G*A
  else: 
    pXbarH = crossComputeKerns(queryX,np.matrix(X[:,2]).T,kernel,symmetric,var)*A
  return pXbarH  
  
# <codecell>

def phi_beta(x, N, n):
    '''
        Input: 
            x: input (observation) vector
            [0..N]: possible values x can take
            n: dimensionality of feature map  
        Output:
            phi: beta-distribution encoding of phi(x)
    '''
    
    p = np.asarray(map(lambda t: (t ** x) * ( (1-t) ** N-x ), 
                       np.linspace(0.05,0.95,n).tolist()));
    return p / sum(p);
    
# <codecell>
    
def returnKernel(kernel, var=1):
  """
  Utility function to return the correct kernel function given a string 
  identifying the name of the kernel
  """
  kernel = kernel.lower()
  if kernel in ['gaussian','normal','l2']:
    kern = lambda XX: np.exp(-pdist2(XX,'sqeuclidean')/var)
  elif kernel in ['laplace','laplacian','l1']:
    kern = lambda XX:  np.exp(-pdist2(XX,'minkowski',1)/var)
  elif kernel in ['dirac', 'delta','kronecker']:
    equals = lambda u,v: 1 - (np.array(u)==np.array(v)).all()
    kern = lambda XX: pdist2(XX,equals)
  elif kernel in ['mahalanobis']:
    VInv = np.inv(var)
    kern = lambda XX: np.exp(-np.power(pdist2(XX,'mahalanobis',VInv),2))
  elif kernel in ['beta']:
    dot = lambda XX: XX.dot(XX)
    kern = lambda XX: dot(phi_beta(XX,var[0],var[1]))
  return kern

# <codecell>
  
def crossComputeKerns(X,Y,kernel,symmetric,var=1):
  """
  Compute pairwise kernel between points in two matrices
  Inputs:
     X: m x D matrix with m samples of dimension D 
     kernel: (string) name of kernel being computed
     Y: n x D matrix with n samples of dimension D 
     kernel: (string) name of kernel being computed
     k: (int) rank of model (number of hidden states)
     symmetric: whether the model has symmetric views or not (should be false
          for HMM)
     var: for gaussian kernel, the variance (sigma^2)
      for laplacian kernel, the bandwidth
      for mahalanobis kernel, the covariance matrix
      for delta kernel, None
  Outputs: tuple (K,L,G), where:
     K: cross-kernel matrix c of dimension m x n
  """
  kernel = kernel.lower()
  if kernel in ['gaussian','normal','l2']:
    kern = lambda XX,YY: np.exp(-cdist(XX,YY,'sqeuclidean')/var)
  elif kernel in ['laplace','laplacian','l1']:
    kern = lambda XX,YY:  np.exp(-cdist(XX,YY,'minkowski',1)/var)
  elif kernel in ['dirac', 'delta','kronecker']:
    equals = lambda u,v: 1 - (np.array(u)==np.array(v)).all()
    kern = lambda XX,YY: cdist(XX,YY,equals)
  elif kernel in ['mahalanobis']:
    VInv = np.inv(var)
    kern = lambda XX,YY: np.exp(-np.power(cdist(XX,YY,'mahalanobis',VInv),2))
  elif kernel in ['beta','betadot']:
    kern = lambda XX,YY: phi_beta(XX,var[0],var[1]).T.dot(phi_beta(YY,var[0],var[1]))
  K = kern(X,Y)   
  return K

# <codecell>
    
def computeKerns(X,kernel,symmetric, var=1):
  """
  Compute pairwise kernels between D-dimensional points arranged into a matrix
  Inputs:
     X: m x D matrix with m samples of dimension D 
     kernel: (string) name of kernel being computed
     k: (int) rank of model (number of hidden states)
     symmetric: whether the model has symmetric views or not (should be false
          for HMM)
     var: for gaussian kernel, the variance (sigma^2)
      for laplacian kernel, the bandwidth
      for mahalanobis kernel, the covariance matrix
      for delta kernel, None
  Outputs: tuple (K,L,G), where:
     K: kernel matrix of first view
     L: kernel matrix of second view
     G: kernel matrix of third view
  """
  kern = returnKernel(kernel,var)
  K = kern(X)   
  if symmetric:
    return (K[:-1,:-1], K[1:,1:], None)
  else:
    return (K[:-2,:-2],K[1:-1,1:-1],K[2:,2:])

# <codecell>
    
def computeKerns3(X,kernel,symmetric, var=1):
  """
  Compute the kernel for samples from 3 views
  Inputs:
     X: D x m matrix with m samples and 3 views
     kernel: (string) name of kernel being computed
     k: (int) rank of model (number of hidden states)
     symmetric: whether the model has symmetric views or not (should be false
          for HMM)
     var: for gaussian kernel, the variance (sigma^2)
      for laplacian kernel, the bandwidth
      for mahalanobis kernel, the covariance matrix
      for delta kernel, None
  Outputs: tuple (K,L,G), where:
     K: kernel matrix of first view
     L: kernel matrix of second view
     G: kernel matrix of third view
  """
  kern = returnKernel(kernel,var)
  X0,X1,X2 = np.matrix(X[:,0]).T,np.matrix(X[:,1]).T, np.matrix(X[:,2]).T
  return (kern(X0),kern(X1),kern(X2))

# <codecell>
  
def kernSpecAsymm(K,L,G,k,view=2,lambda0=1e-2):
  """
  Algorithm 1 from Song et al. (2014). adapted to asymmetric view
  Inputs:
  *K: kernel matrix from view 1 of shape (m x m)
  *L: kernel matrix from view 2 of shape (m x m)
  *G: kernel matrix from view 3 of shape (m x m)
  *k: desired rank (number of hidden states)
  *view: (int) which view to learn (either 1,2, or 3)
  *lambda0: (float) regularization parameter
  Outputs:
  *A: matrix of shape m x k
  *pi: vector of shape k
  """
  K,L,G = np.matrix(K), np.matrix(L), np.matrix(G)
  m = K.shape[1]  # number of samples per view
  if view==2:
    S, beta = sortedEig(K*G*K/(m**2),K,k,lambda0)  
    t1 = np.matrix(beta.real)*np.matrix(np.diag(np.power(S,-0.5)).real)
    S, beta = sortedEig(G*K*G/(m**2),G,k,lambda0)#Lnk = L*np.matrix(sortedEig(L*K*L,L,k)[1].real)
    t2 = np.matrix(beta.real)*np.matrix(np.diag(np.power(S,-0.5)).real)
    Knk = K*t1; Gnk = G*t2
    H = Gnk*np.linalg.inv(Knk.T*Gnk)*Knk.T  #NEEDS TO BE VERIFIED
    (S,beta) = sortedEig(L*H.T*L*H*L/(m**2),L,k,lambda0) #find generalized eigenvectors
    S,beta=S.real,np.matrix(beta.real)
    Sroot = np.matrix(np.diag(np.power(S,-0.5))) 
    term1 = L*beta*Sroot
    T = trilinear('I', H.T*term1,term1, H*term1)/m  
  elif view==3:
    S, beta = sortedEig(K*L*K/(m**2),K,k,lambda0)
    t1 = np.matrix(beta.real)*np.matrix(np.diag(np.power(S,-0.5)).real)
    S, beta = sortedEig(L*K*L/(m**2),L,k,lambda0)#Lnk = L*np.matrix(sortedEig(L*K*L,L,k)[1].real)
    t2 = np.matrix(beta.real)*np.matrix(np.diag(np.power(S,-0.5)).real)  
    Knk = K*t1; Lnk = L*t2  
    H = Knk*np.linalg.inv(Lnk.T*Knk)*Lnk.T #Symmetrization matrix
    (S,beta) = sortedEig(G*H.T*G*H*G/(m**2),G,k,lambda0) #find generalized eigenvectors
    S,beta=S.real,np.matrix(beta.real)
    Sroot = np.matrix(np.diag(np.power(S,-0.5))) 
    term1 = G*beta*Sroot
    T = trilinear('I', H*term1,H.T*term1,term1)/m
  (M,lambda0) = tentopy.eig(T,inner,outer) 
  M = np.matrix(M[:,:k])
  lambda0 = np.array(lambda0[:k]).flatten()
  A = beta*Sroot*M*np.diag(lambda0)
  pi = np.power(lambda0,-2).T
  return (A,pi)

# <codecell>

def sortedEig(X,M=None,k=None, lambda0=0):
  """
    Return the k largest eigenvalues and the corresponding eigenvectors of the
    solution to X*u = b*M*u
    Inputs:
      X: matrix
      M: matrix
      k: (int) if k is None, return all but one.
      lambda0: (float) regularization parameter to ensure positive definiteness
            so that cholesky decomposition works
    Outputs:
     b: vetor of eigenvalues
     U: matrix of eigenvectors; each column is an eigenvector
  """
  if k is None:
    k = X.shape[0]
    if M is None:
      (b,U) = scipy.linalg.eig(X)
    else:
      (b,U) = scipy.linalg.eig(X,M+lambda0*np.eye(M.shape[0]))
    idx = b.argsort()[-k:][::-1]
    return b[idx], U[:,idx]
  else:
    if M is None:
        (b,U) = scipy.sparse.linalg.eigsh(X,k)
    else:
        (b,U) = scipy.sparse.linalg.eigsh(X,k,M+lambda0*np.eye(M.shape[0]))
  return b,U

# <codecell>
  
def kernSpecSymm(K,L,k):
  """
  Kernel Spectral Algorithm (Algorithm 1 from Song et al. (2014)) for symmetric
  views.  For now, just use the asymmetric views algorithm, since this is just
  a special case of that algorithm.
  Inputs: 
    *K, L: kernel matrices as defined in Sec 5.2 of Song et al. (2014)
    *k: desired rank of model
  Outputs:
    *pi: vector of prior probabilities
    *A: matrix
  """
  m = K.shape[1]/3
  (S, beta) = scipy.linalg.eig(K*L*K,K)
  S = np.diag(S[:k])  
  beta = beta[:,:k]
  qq = np.size(np.power(S,-0.5)*beta.T*K[:,1])
  term1 = np.power(S,-0.5)*beta.T
  T = tentopy.tensor_outer(np.zeros(qq),3)
  for kk in range(k):
    chi_1 = term1*K[:,kk]
    chi_2 = term1*K[:,m+kk]
    chi_3 = term1*K[:,2*m+kk]
    T += symmetricTensor(chi_1,chi_2,chi_3)
  (M, lambda0) = tentopy.eig(T,inner,outer)
  A = beta*np.power(S,-0.5)*M*np.diag(lambda0)
  pi = np.power(lambda0,-2).T
  return (A,pi)
  
# <codecell>

def trilinear(T,W1,W2,W3):
  """
  Compute trilinear form T(W1,W2,W3).  If T=='I', it is taken to be the third-
  order identity tensor I such that I[i,j,k] = \delta_{(i==j==k)}.
  Inputs:
    T: 3rd-order tensor of shape (N_T x N_T x N_T)
    W1,W2,W3: each Wi is a vector or matrix of shape (N_T x N_i)
  Returns:
    X3: Trilinear form of shape (N_1 x N_2 x N_3)
  """
  def matrix(W):
    if len(W.shape)==1:
      return np.matrix(W).T
    else:
      return W
  W1,W2,W3 = matrix(W1),matrix(W2),matrix(W3)
  N1 = xrange(W1.shape[1])
  N2 = xrange(W2.shape[1])
  N3 = xrange(W3.shape[1])
  NT = xrange(W1.shape[0])
  X3 = tentopy.tensor_outer(np.zeros(len(N1)), 3)
  # TODO: figure out the equivalent numpy routines
  if type(T) is str:
    if T == 'I':
        for (i1,i2,i3,j) in itertools.product(N1,N2,N3,NT):
            X3[i1,i2,i3] += W1[j,i1] * W2[j,i2] * W3[j,i3]
  else:  
    for (i1,i2,i3,j1,j2,j3) in itertools.product(N1,N2,N3,NT,NT,NT):
      X3[i1,i2,i3] += T[j1,j2,j3] * W1[j1,i1] * W2[j2,i2] * W3[j3,i3]
  return X3

# <codecell>

def medianTrick(X,kernel='gaussian'):
    """
    Implementation of the median trick for bandwidth selection from Gretton et
    al (2006) NIPS paper.
    Set var = 0.5*(median(|X_i-X_j|)[:])**2,i != j (TODO: check this condition)
    """
    if kernel=='gaussian':
        return 0.5*np.median(pdist(X,'sqeuclidean'))

# <codecell>

def symmetricTensor(a,b,c):
  """
  symmetric tensor product from section 5.2 of Song et al. (2014)
  """
  term1 = np.tensordot(np.tensordot(a,b,0),c,0)
  term2 = np.tensordot(np.tensordot(c,a,0),b,0)
  term3 = np.tensordot(np.tensordot(b,c,0),a,0)
  return term1+term2+term3
