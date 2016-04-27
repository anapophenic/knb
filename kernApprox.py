# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:52:16 2016
Code for computing the finite-dimensional kernel approximations developed by 
Vedaldi & Zisserman (2010; 2011) for homogeneous histogram kernels

References:
A. Vedaldi and A. Zisserman. (2010).  Efficient Additive Kernels via Explicit
   Feature Maps.  Proceedings of CVPR.
   
A. Vedaldi and A. Zisserman. (2011).  Efficient Additive Kernels via Explicit
   Feature Maps.  Pattern Analysis and Machine Intelligence.

"""
import numpy as np

def return_kappa(kernel):
    """
    Returns the kernel signature function kappa associated with a homogeneous
    kernel.  Taken from the table in Vedaldi & Zisserman 2011, Fig. 1    
    """
    if kernel.lower() in ['hellinger','hellingers']:
        return lambda x: x==0
    elif kernel.lower() in ['chi square','chi2','chi squared']:
        return lambda x: 1./np.cosh(x*np.pi)
    elif kernel.lower() in ['intersection','intersect']:
        return lambda x: 2./(np.pi*(1+4*(x**2)))
    elif kernel.lower() in ['jensen-shannon','js','jensen shannon','jsd']:
        return lambda x: 2./(np.log(4)*np.cosh(np.pi*x)*(1+4*x))

def kernApprox(kappa, n, L,x):
    psi = np.zeros(n*L)
    psineg = np.zeros(n*L)
    psi[0] = np.sqrt(x*kappa(0))
    for j in np.arange(1,n*L-1,2):
        t1 = np.sqrt(2*x*kappa((j+1)/2.))
        t2 = (j+1)/2.*L*np.log(x)
        psi[j] = t1*math.cosine(t2)
        psi[j+1] = t1*math.sine(t2)
        t1 = np.sqrt(2*x*kappa(-(j+1)/2.))
        psineg[-j] = t1*math.cosine(-t2)
        psineg[-j-1] = t1*math.sine(-t2)
    return np.concatenate((psineg[1:], psi))