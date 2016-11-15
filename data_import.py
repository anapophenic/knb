import scipy.io
import numpy as np
import binom_hmm as bh
from collections import Counter

def importance_weightify(x_zipped, l):
    return Counter(x_zipped[:l]);

def load_from_file(filename):
    mat = scipy.io.loadmat(filename)
    coverage = mat['h']
    methylated = mat['mc']

    print 'shape of the original sequence = '
    print np.shape(coverage)

    return coverage, methylated

def group_horizontal(seqs, dims):
    return np.sum(seqs[:,dims], axis=1)

#def group_vertical(seqs, dims):
#    return np.sum(seqs[dims,:], axis=0)

def group_vertical(seqs, s):
    l, r = np.shape(seqs)
    grouped = sum([seqs[i:l-s+i+1,:] for i in range(s)])
    return grouped[range(0,l-s+1,s),:]

def seq_prep(filename, l=None, s=1, ctxt_group=[range(16)]):
    '''
    Suppose the data has contexts x,y, then the data is a 2d array of size r * l
    '''
    coverage, methylated = load_from_file(filename);

    if l is None:
        l = len(coverage)
    r = len(ctxt_group)
    l, dim = np.shape(coverage)

    #vertical_groups = [xrange(i*s,(i+1)*s) for i in range(l/s)]
    #coverage_h = np.array([group_vertical(coverage, vertical_groups[j]) for j in range(l/s)])
    #methylated_h = np.array([group_vertical(methylated, vertical_groups[j]) for j in range(l/s)])

    coverage_h = group_vertical(coverage, s)
    methylated_h = group_vertical(methylated, s)

    #print np.shape(coverage_h)
    #print coverage_h

    coverage_hv = np.array([group_horizontal(coverage_h, ctxt_group[i]) for i in range(r)])
    methylated_hv = np.array([group_horizontal(methylated_h, ctxt_group[i]) for i in range(r)])

    return coverage_hv, methylated_hv


def triples_from_seq(coverage, methylated, formating):
  N = np.amax(coverage)
  print 'N = '
  print N

  #x = map(lambda c, m: bh.to_x(c, m, N), coverage, methylated)
  #print 'coverage = '
  #print np.shape(coverage)
  r, l = np.shape(coverage)
  x = [tuple([bh.to_x(coverage[c,i], methylated[c,i], N) for c in range(r)]) for i in range(l)]

  # compute E[1/(n+2)]
  a = np.sum(1.0 / (coverage+2), axis = 1) / l

  if formating=='explicit':
    x_zipped = zip(x[0:l], x[1:l+1], x[2:l+2])
    return N, x_zipped, a
  else:
    X = np.vstack((x[0:l], x[1:l+1], x[2:l+2])).T
    return N, X, a


#def data_prep(filename, formating='explicit', l=None, s=1, ctxts=[range(16)]):
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

#  coverage, methylated = seq_prep(filename, l, s, ctxts);
#  return triples_from_seq(coverage, methylated, formating) + (coverage, methylated)



if __name__ == '__main__':

  filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chrY_binsize100.mat'

  N, X, a = data_prep(filename, ctxt = range(12,16));
  print N
  print X
  print a


  #print mat
  #print mat['mc']
  #print mat['h']
  #print mat['bins']
  #coverage = [[sum(sum(coverage[lims[i]:lims[i+1], ctxts[c]])) for c in range(r)] for i in range(l_n)]
  #methylated = [[sum(sum(methylated[lims[i]:lims[i+1], ctxts[c]])) for c in range(r)] for i in range(l_n)]


  #x_zipped_listed_iw =
  #l = np.shape(X_zipped)[0]
  #print x_zipped

  #print 'length of the grouped sequence = '
  #print len(coverage)

#X0 = np.zeros(l)
#for i in range(l):
#  X0[i] = coverage[i]*(N+1) + methylated[i]
