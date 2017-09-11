import numpy as np
from pyglrm import glrm, nnfm, pca, rpca

"""usage example"""
if __name__ == "__main__":
    
    # form an n x d array
    n = 10 # number of examples
    d = 5  # number of dimensions
    A = np.arange(n*d).reshape(n, d)
    k = 2
    
#    g = glrm() #PCA
    g = pca() #PCA
#    g = nnmf() #NNMF
#    g = rpca() #RPCA

    W = g.dimension_reduce(A, k)
    print(W)
    
    #calculate output map with respect to a new vector
    v = np.arange(d)
    r = g.output_map(v)
    print(r)
