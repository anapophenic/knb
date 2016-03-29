# knb
Kernel multi-view spectral learning algorithm for mixture models/HMM from Song et al. (2014) ICML paper

The main code is in kernelNaiveBayes.py, but you need to have tentopy.py in your path also for it to run.

To test it out and see how it works, see the example in knbTest.py where we create synthetic data from a 2-component GMM and plot the estimates of the distributions p(x|h) learned from the data.  The plot is saved to "pdf.png" in the current working directory.

Dependencies (all of them are included in the standard Anaconda distribution of Python):
*numpy
*scipy
*matplotlib (for plotting)
*math
*itertools
