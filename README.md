# knb
Kernel multi-view spectral learning algorithm for mixture models/HMM using kernelized method presented in Song et al. (2014) ICML paper.  Code is still in alpha stage.  It is statistically correct, but needs to be documented and some inelegant hacks need to be removed.  If you have questions or would like help using it, email me at edg@icsi.berkeley.edu (but it might take me some time to get back to you).

The main code is in kernelNaiveBayes.py (function kernXMM), but you need to have tentopy.py in your path also for it to run. 

To test it out and see how it works, see the example in knbTest.py where we create synthetic data from a 2-component GMM and plot the estimates of the distributions p(x|h) learned from the data.  The plot is saved to "pdf.png" in the current working directory.

Dependencies (all of them are included in the standard Anaconda distribution of Python):
*numpy
*scipy
*matplotlib (for plotting)
*math
*itertools
