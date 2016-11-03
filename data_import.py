import scipy.io
import numpy as np
import binom_hmm as bh
from collections import Counter

def group(coverage, methylated, s):

  l = np.shape(coverage)[0]

  # zipping this way group every s elts together
  coverage_zipped = zip(*[coverage[i:(l-s+i)] for i in range(s)])
  methylated_zipped = zip(*[methylated[i:(l-s+i)] for i in range(s)])
  coverage_merged = map(sum, coverage_zipped)
  methylated_merged = map(sum, methylated_zipped)
  coverage_subs = coverage_merged[0:l-s+i:s]
  methylated_subs = methylated_merged[0:l-s+i:s]

  return coverage, methylated

def importance_weightify(x_zipped, l):
    # preparing data
    #l = np.shape(X_zipped)[0]
    return Counter(x_zipped[:l])

def load_from_file(filename):
    mat = scipy.io.loadmat(filename)
    #print mat
    #print mat['mc']
    #print mat['h']
    #print mat['bins']

    coverage = mat['h']
    methylated = mat['mc']

    print 'length of the original sequence = '
    print len(coverage)

    return coverage, methylated

def select_subseq(coverage, methylated, l, ctxt):
    coverage = coverage[:l, ctxt]
    methylated = methylated[:l, ctxt]

    print 'shape of the coverage parameter = '
    print np.shape(coverage)

    # merging all the contexts with cytosine(C) at this point:
    coverage = np.sum(coverage, axis=1);
    methylated = np.sum(methylated, axis=1);

    return coverage, methylated

def seq_prep(filename, l=None, s=1, ctxts=[range(16)]):
    (coverage, methylated) = load_from_file(filename);
    if l is None:
        l = len(coverage)

    r = len(ctxts)
    coverage = []
    methylated = []

    for ctxt in ctxts:
        coverage_c, methylated_c = select_subseq(coverage, methylated, l, ctxt);
        # merge every s observations
        coverage_c, methylated_c = group(coverage_c, methylated_c, s);
        coverage.append(coverage_c);
        methylated.append(methylated_c);

    coverage = np.asarray(coverage).T.tolist()
    methylated = np.asarray(methylated).T.tolist()
    #print 'length of the grouped sequence = '
    #print len(coverage)
    #Suppose the data has contexts x,y, then the data is like
    #[[x1, y1], [x2, y2], .... [xn, yn]]

    return coverage, methylated


def triples_from_seq(coverage, methylated, formating):
  N = np.amax(coverage)

  print 'N = '
  print N

  #x = map(lambda c, m: bh.to_x(c, m, N), coverage, methylated)
  l, r = np.shape(coverage)
  x = [[bh.to_x(coverage[i,j], methylated[i,j], N) for j in range(r)] for i in range(l)]

  l = len(coverage)
  # compute E[1/(n+2)]
  a = sum(1.0 / (coverage+2)) / l

  #X0 = np.zeros(l)
  #for i in range(l):
  #  X0[i] = coverage[i]*(N+1) + methylated[i]

  if formating=='explicit':
    x_zipped = zip(x[0:l], x[1:l+1], x[2:l+2])
    return N, x_zipped, a
  else:
    X = np.vstack((x[0:l], x[1:l+1], x[2:l+2])).T
    return N, X, a


def data_prep(filename, formating='explicit', l=None, s=1, ctxts=[range(16)]):
  """
  Main function for importing triples from raw INTACT DNA methylation data
  Inputs:
   filename: filename for INTACT DNA methylation data
   format: whether to return the data formatted for explicit feature map
      or for kernel feature map
   l: maximum length of data to sample; if None, the whole set is sampled
   s: number of methylations / coverages to merge
  Outputs:
   N: number of maximum number of coverage
   X_importance_weighted: a dictionary of
    key: triples (x_1, x_2, x_3)
    value: total number of cooccurrence of (x_1, x_2, x_3) (importance weight)
    a: correction term \E[1/(n+2)] used in explicit feature map
  """

  coverage, methylated = seq_prep(filename, l, s, ctxts);
  return triples_from_seq(coverage, methylated, formating) + (coverage, methylated)



if __name__ == '__main__':

  filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chrY_binsize100.mat'

  N, X, a = data_prep(filename, ctxt = range(12,16));
  print N
  print X
  print a
