"""
This file implements a test for dimensionality reduction methods.
A method that complies with the dimensionality reduction API will pass this test.
"""
import numpy as np

"""
test_dim_reducer: tests dimensionality reduction method

Input:
  MyDimReducer: an instance of a class implementing the DimReducer interface

Output:
  W: Dimensionality reduced vectors. Here, we output a numpy matrix with one row per vector.

"""
def test_dim_reducer(MyDimReducer):
    # form an n x d array
    n = 10 # number of examples
    d = 5  # number of dimensions
    A = np.arange(n*d).reshape(n, d)
    k = 2

    try:
        # dimensionality reduction method implemented
        W = MyDimReducer.dimension_reduce(A, k)
        # W has n rows and k columns
        W[n-1,k-1]
        # MyDimReducer has an output map
        hasattr(MyDimReducer, "output_map")
        hasattr(MyDimReducer, "is_dimensionality_reduction")
        assert isinstance(MyDimReducer.is_dimensionality_reduction, bool)
        hasattr(MyDimReducer, "hyperparameters")
        assert isinstance(MyDimReducer.hyperparameters, dict)

    except:
        print("Dimensionality reduction method does not comply with API:", sys.exc_info()[0])
        raise

    print("All tests pass")

"""usage example"""
if __name__ == "__main__":

    from DimReducer import DimReducer
    dr = DimReducer()
    test_dim_reducer(dr)
