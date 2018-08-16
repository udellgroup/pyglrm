"""
This class implements a trivial dimensionality reduction method:
It keeps the first k entries of every vector as the dimension-reduced feature.
It is designed to allow TA2 methods to verify they handle Dimensionality Reduction methods correctly,
and to provide a template for TA1 performers to follow to implement dimensionality reduction methods.

Input:
  A: Collection of vectors in high dimensional space. Concretely, inputs are duck-typed: they are any doubly-indexable object that can be called as A[i,j]. Rows i are samples and columns j are features.
  k: Dimensionality of output space

Output:
  W: Dimensionality reduced vectors. Here, we output a numpy matrix with one row per vector.

"""
class DimReducer:

  is_dimensionality_reduction = True
  hyperparameters = {}

  def __init__(self):
    return None

  def dimension_reduce(self, A, k):

    # define output map
    self.output_map = lambda v: v[:k]

    # reduce data
    return A[:,:k]

"""usage example"""
if __name__ == "__main__":

    import numpy as np
    # form an n x d array
    n = 10 # number of examples
    d = 5  # number of dimensions
    A = np.arange(n*d).reshape(n, d)
    k = 2
    from DimReducer import DimReducer
    dr = DimReducer()
    W = dr.dimension_reduce(A, k)
