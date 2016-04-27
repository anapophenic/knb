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
import scipy.stats

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

def plot2(pXbarH,xRange,savepath='pdf.png'):
  # plot results
  fig = plt.figure(figsize=(8,4))
  ax = fig.add_subplot(1,2,1); x = xRange; y = np.array(pXbarH[:,0].T).flatten()
  ax.scatter(x,y); ax.set_title('pdf of component h=0'); ax.set_xlabel("x")
  ax.set_ylabel("pdf"); ax = fig.add_subplot(1,2,2); x = xRange; y = np.array(pXbarH[:,1].T).flatten()
  ax.scatter(x,y)
  ax.set_title('pdf of component h=1'); ax.set_xlabel("x"); ax.set_ylabel("pdf")
  fig.savefig(savepath)

def plot3(pXbarH,xRange,truepdf,savepath='pdf.png'):
  # plot results
  fig = plt.figure(figsize=(8,4))
  ax = fig.add_subplot(1,2,1); x = xRange; y = np.array(pXbarH[:,0].T).flatten()
  ax.plot(x,y);
  y = np.array(truepdf[:,0].T).flatten(); ax.plot(x,y)
  ax.set_title('pdf of component h=0'); ax.set_xlabel("x")
  ax.set_ylabel("pdf"); ax = fig.add_subplot(1,2,2); x = xRange; y = np.array(pXbarH[:,1].T).flatten()
  ax.plot(x,y)
  y = np.array(truepdf[:,1].T).flatten(); ax.plot(x,y)
  ax.set_title('pdf of component h=1'); ax.set_xlabel("x"); ax.set_ylabel("pdf")
  fig.savefig(savepath)

def testCode():
# Example test code
  k=2; #number of latent states
  pXbarH = {}
  randinits = 100 #number of outer loop iterations (restarts) to run
  for i in range(randinits):
    X,H,means,variances = initGaussMM(k,1000,means=[1.5,-1],variances=[0.4,0.4])
    xRange = np.matrix(np.linspace(min(means)-2*max(variances)**0.5,
                                     max(means)+2*max(variances)**0.5,500)).T #create 500 equally spaced samples for visualizing p(x|h)
    pXbarH[i] = knb.kernXMM(X,k,xRange,var=.01) #compute p(x|h) estimate
    
  sumprob = np.matrix(np.zeros(np.shape(pXbarH[0])))
  for i in pXbarH.keys():
    map0 = xRange[np.argmax(pXbarH[i][:,0])]
    if np.abs(map0-means[0])<np.abs(map0-means[1]):
        sumprob +=pXbarH[i]
    else:
        for j in range(pXbarH[i].shape[0]):
          sumprob[j,0] += max([0,np.matrix(pXbarH[i][j,1])])
          sumprob[j,1] += max([0,np.matrix(pXbarH[i][j,0])])

  #scale the pdfs to make them comparable

  truepdf = np.zeros(pXbarH[0].shape)
  for kk in range(k):
    for i,item in enumerate(xRange):
      truepdf[i,kk] = scipy.stats.norm.pdf(item, means[kk],variances[kk])
    
for kk in range(k):
    o = truepdf[:,kk].sum()
    p = sumprob[:,kk].sum()
    for i in range(len(xRange)):
        truepdf[i,kk] = truepdf[i,kk]/o
        sumprob[i,kk] = sumprob[i,kk]/p
plot3(sumprob,xRange,truepdf,savepath='pdfcompare.png')

    
plot2(sumprob,xRange,savepath='pdf.png')


#maximum a posteriori estimate of component 0:
map0 = xRange[np.argmax(pXbarH[:,0])]
#maximum a posterior estimate of component 1:
map1 = xRange[np.argmax(pXbarH[:,1])]

print 'Maximum a posteriori means:\t'+str(map0)+'\t'+str(map1)

########################################################

########################################################
import kernelNaiveBayes as knb
import data_import as d_i
import moments_cons as m_c
basepath = 'C:/Users/e4gutier/Downloads/mukamel/cndd/emukamel/HMM/Data/Binned/'



l=5000
N,X,a = d_i.data_prep(basepath+'allc_AM_E1_chrY_binsize100.mat','kern',l)
kernel = 'beta_shifted'

N,X_importance_weighted,a = d_i.data_prep(basepath+'allc_AM_E1_chrY_binsize100.mat','explicit',l)

#noise = np.random.rand(Data.shape[0],Data.shape[1])/1e3 #jiggle the data points a little bit to improve conditioning
m = 3; n=5 #m is number of hidden states; n is number of bins in kernel
xRange = np.arange(0,X.max(),int(X.max()/n))
O_h,T_h = knb.kernXMM(X,m,xRange,var=[N,n],kernel='beta_shifted')
O_h,T_h = m_c.col_normalize(O_h),m_c.col_normalize(T_h)

##
kernel = 'beta_shifted'; var = [N,n];k=m; queryX = xRange
(K,L,G) = knb.computeKerns3(X,kernel,False,var)
(A,pi) = knb.kernSpecAsymm(K,L,G,k,view=2)
(A3,_) = knb.kernSpecAsymm(K,L,G,k,view=3)
O_h = knb.crossComputeKerns(queryX,X[:,2].T,kernel,False,var)*A
OT_h = knb.crossComputeKerns(queryX,X[:,2].T,kernel,False,var)*A3
T_h = np.linalg.pinv(O_h).dot(OT_h)
##

phi = m_c.phi_beta_shifted;
P_21, P_31, P_23, P_13, P_123 = m_c.moments_cons_importance_weighted(X_importance_weighted, phi, N, n);
    
C_h, T_h = m_c.estimate(P_21, P_31, P_23, P_13, P_123, m)

C_h_p, T_h_p = m_c.estimate_refine(C_h, P_21, phi, N, n, m, a)
