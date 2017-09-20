import numpy as np
from pyglrm import *
    
A = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 5, 6, 7]])
k = 2
losses = QuadLoss()
rx = ZeroReg()
ry = ZeroReg()
g = glrm() #create a class for GLRM (Here it does PCA)
X, Y, ch = g.fit_transform(A, k)
A_pca = np.dot(np.transpose(X), Y) #result for PCA
a_new = np.array([6, 7, 8, 9]) #initialize a new row to be tested
x = g.predict(a_new) #get the latent representation of a_new
