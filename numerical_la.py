import numpy as np
'''
W_1: k x n matrix
T: k x k x k tensor
Return: n x n x n tensor
'''
def fast_trilinear(T, W1, W2, W3):
  NT = W1.shape[0];

  if type(T) is str:
    if T == 'I':
        M = np.zeros((NT, NT, NT));
    	for i in xrange(NT):
		M[i,i,i] = 1
  else:
    M = T

  M1 = ttm(M, W1, 0)
  M12 = ttm(M1, W2, 1)
  M123 = ttm(M12, W3, 2)

  return M123

'''
Tensor Matrix multiplication
Dimensionally stable (the modes are kept in the same positions)
'''
def ttm(M, W, axis):
  dim = len(np.shape(M))
  T = np.tensordot(M, W, axes = ([axis], [0]))
  rear_dims = tuple(range(0, axis) + [dim-1] + range(axis,dim-1))
  T = np.transpose(T, rear_dims)
  return T
