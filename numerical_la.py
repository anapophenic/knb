import numpy as np

def fast_trilinear(T, W1, W2, W3):
  NT = W1.shape[0];

  # Generate T is it is implicitly given
  if type(T) is str:
    if T == 'I':
        M = np.zeros((NT, NT, NT));
    	for i in xrange(NT):
		M[i,i,i] = 1
  else:
    M = T

  M1 = dim_stable_tpm(M, W1, 0)
  M12 = dim_stable_tpm(M1, W2, 1)
  M123 = dim_stable_tpm(M12, W3, 2)

  return M123

# <codecell>

def dim_stable_tpm(M, W, axis):
  dim = len(np.shape(M))
  T = np.tensordot(M, W, axes = ([axis], [0]))
  rear_dims = tuple(range(0, axis) + [dim-1] + range(axis,dim-1))
  T = np.transpose(T, rear_dims)
  return T
