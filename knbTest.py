###############################################################################
# knbTest.py
#
# Example usage of implementation of kernel multi-view spectral algorithm of 
#  Song et al. (2014)
#
# Author: E.D. Gutierrez (edg@icsi.berkeley.edu)
# Created: 24 March 2016
# Last modified: 29 March 2016
#
###############################################################################

import kernelNaiveBayes as knb
import numpy as np
import scipy
import scipy.spatial.distance
import matplotlib.pyplot as plt

def initGaussMM(k,m, means=None,variances=None):
  """
  Create synthetic data drawn from gaussian mixture model.
  Inputs:
    k: number of hidden states (mixture components)
    m: number of samples per view
  Optional:
    means: array of k means of mixture components
    variances: array of k variances of mixture ocmponents
  Return:
    X: array of shape (m x 3 ) with m observations for each of 3 views
    H: array of shape (m x 1) listing mixture component that generate each 
     observation
    means: array of k means of mixture components
    variances: array of k variances of mixture ocmponents
  """
  X = np.zeros((m,3))
  H = np.zeros(m,int)
  if (means is None) or (variances is None):
    means = []
    variances = []
    for kk in range(k):
      means.append(float(kk))
      variances.append(0.1*(kk+1))
  for mm in range(m):
    H[mm] = int(np.random.randint(0,k))
    for j in range(3):
      X[mm,j] = np.random.normal(means[H[mm]],variances[H[mm]])
  return X,H,means,variances

if False:
    k=2; 
    X,H,means,variances = initGaussMM(k,1000)
    xRange = np.matrix(np.linspace(-2,3,500)).T #create 500 equally spaced samples in range [-2,3] for visualizing p(x|h)
    pXbarH = knb.kernXMM(X,k,xRange,var=0.1) #compute p(x|h) estimate
    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    m=1000;x = X[:m,2]; y = np.array(pXbarH[:m,0].T).flatten()*10e23
    ax.scatter(x,y)
    ax.set_title('scaled prob density of component h=0')
    ax.set_xlabel("x")
    ax.set_ylabel("scaled pdf")
    ax = fig.add_subplot(1,2,2)
    m=1000;x = X[:m,2]; y = np.array(pXbarH[:m,1].T).flatten()*10e23
    ax.scatter(x,y)
    ax.set_title('scaled prob density of component h=0')
    ax.set_xlabel("x")
    ax.set_ylabel("scaled pdf")
    fig.savefig("pdf.png")
