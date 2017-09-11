import julia
import numpy as np

j = julia.Julia()
j.using("LowRankModels")

#Losses
def QuadLoss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, float):
        raise TypeError
    return j.QuadLoss(scale, domain)

def L1Loss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, float):
        raise TypeError
    return j.L1Loss(scale, domain)

def HuberLoss(scale=1.0, domain=j.RealDomain(), crossover=1.0):
    if not isinstance(scale, float) or not isinstance(crossover, float):
        raise TypeError
    return j.HuberLoss(scale, domain, crossover)


#Regularizers
def ZeroReg():
    return j.ZeroReg()

def NonNegOneReg(scale=1):
    return j.NonNegOneReg(scale)

def QuadReg(scale=1):
    return j.QuadReg(scale)

def NonNegConstraint():
    return j.NonNegConstraint()


class glrm:
    
    is_dimensionality_reduction = True
    
    #class constructor: default being PCA
    def __init__(self, losses = QuadLoss(), rx = ZeroReg(), ry = ZeroReg()):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}


    def dimension_reduce(self, A, k):
            
            glrm_j = j.GLRM(A, self.losses, self.rx, self.ry, k)
            X, Y, ch = j.fit_b(glrm_j)
            self.Y = Y
            self.k = k
            
            return np.dot(np.transpose(X), Y)
    
    
    def output_map(self, v): #calculate the output map; requires the input to be an numpy array
        
        v = v.reshape(1, -1)
        
        
        #make sure dimension_reduce has already been executed beforehand, and Y and v match in terms of numbers of columns
        try:
            self.Y
        except NameError:
            raise Exception('Initial GLRM fitting not executed!')
        else:
            try:
                if self.Y.shape[1] != v.shape[1]:
                    raise ValueError
            except ValueError:
                raise Exception('Dimension of input vector does not match Y!')
            else:
                self.Y = self.Y.astype(float) #make sure column vectors finally have the datatype Array{float64,1} in Julia
                num_cols = self.Y.shape[1]
                ry = [j.FixedLatentFeaturesConstraint(self.Y[:, i]) for i in range(num_cols)]
                glrm_new_j = j.GLRM(v, self.losses, self.rx, ry, self.k)
                x, yp, ch = j.fit_b(glrm_new_j)
                return x


class pca(glrm):
    def check_success(): #may not needed; just to make sure this class is not empty so that it can be correctly defined
        return 0


class nnmf(glrm):
    def __init__(self, losses = QuadLoss(), rx = NonNegConstraint(), ry = NonNegConstraint()):
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}


class rpca(glrm):
    def __init__(self, losses = HuberLoss(), rx = QuadReg(), ry = QuadReg()):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}






