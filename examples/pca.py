import numpy as np
from pyglrm import *
    
A = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 5, 6, 7]])

g = pca(n_components=2) #create a class for PCA
g.set_training_data(inputs=A)
g.fit()
a_new = np.array([6, 7, 8, 9]) #initialize a new row to be tested
x = g.produce(inputs=a_new) #get the latent representation of a_new
