import scipy.io
import numpy as np
from collections import Counter

def data_prep(filename):
  """
  Main function for importing triples from raw INTACT DNA methylation data
  Inputs:
     filename: filename for INTACT DNA methylation data
  Otputs:
     N: number of maximum number of coverage
     X_importance_weighted: a dictionary of 
        key: triples (x_1, x_2, x_3)
        value: total number of cooccurrence of (x_1, x_2, x_3) (importance weight) 
        a: correction term \E[1/(n+2)] used in explicit feature map
  """   
    mat = scipy.io.loadmat(filename)

    #print mat
    #print mat['mc']
    #print mat['h']
    #print mat['bins']

    coverage = mat['h']
    methylated = mat['mc']
    N = np.amax(coverage)
    
    print np.shape(coverage)
    
    # merging all the contexts with cytosine(C) at this point:
    coverage = np.sum(coverage, axis=1);
    methylated = np.sum(methylated, axis=1);
    
    
    
    # preparing data
    l = np.shape(coverage)[0]
    print 'l = '
    print l
    
    X0 = coverage * (N+1) + methylated
    
    # compute E[1/(n+2)]
    a = sum(1.0 / (coverage+2)) / l
    
    #X0 = np.zeros(l)
    #for i in range(l):
    #    X0[i] = coverage[i]*(N+1) + methylated[i]
    
    X_zipped = zip(X0[0:l-2], X0[1:l-1], X0[2:l])
    
    X_importance_weighted = Counter( X_zipped )
    
    return N, X_importance_weighted, a
    
    

if __name__ == '__main__':

    filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chrY_binsize100.mat'
    
    N, X, a = data_prep(filename);
    print N
    print X
    print a
