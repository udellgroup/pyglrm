import julia
import numpy as np
import pandas as pd

j = julia.Julia()
j.using("LowRankModels")

__all__ = ['QuadLoss','L1Loss','HuberLoss','ZeroReg','NonNegOneReg','QuadReg','NonNegConstraint','glrm', 'pca', 'nnmf', 'rpca', 'observations']

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

#the Python glrm class
class glrm:
    is_dimensionality_reduction = True
    is_feature_selection = False
    
    #class constructor: default being PCA
    def __init__(self, losses = QuadLoss(), rx = ZeroReg(), ry = ZeroReg(), n_components=2, X=None, Y=None, offset=False, scale=False):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.k = n_components
        self.training_inputs = None
        self.index = None
        self.fitted = False
        self.offset = offset
        self.scale = scale
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}
        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y
    

#fit the dimensionality reduction method to the input and then output dimensionality-reduced results

    def fit(self, inputs=None):
        if self.fitted and inputs is None:
            return
        if inputs is None and self.training_inputs is None:
            raise ValueError("Missing training data.")
        elif inputs is not None:
            if type(inputs) is np.ndarray:
                self.training_inputs = inputs
            elif type(inputs) is pd.core.frame.DataFrame:
                self.training_inputs = inputs.values
                self.index = list(inputs.index)
            else:
                raise TypeError("Input must be either numpy array or pandas DataFrame!")
        columns_missing = np.where(np.array(np.sum(np.invert(np.isnan(self.training_inputs)), axis=0))==0)[0]
        self.training_inputs = np.delete(self.training_inputs, columns_missing, 1)
        obs = _observations(self.training_inputs)
        glrm_j = j.GLRM(self.training_inputs, self.losses, self.rx, self.ry, self.k, obs=obs, offset=self.offset, scale=self.scale)
        X, Y, ch = j.fit_b(glrm_j)
        self.X = X
        self.Y = Y
        self.fitted = True


    def fit_transform(self, inputs=None):
        if self.fitted and inputs is None:
            return
        if inputs is None and self.training_inputs is None:
            raise ValueError("Missing training data.")
        elif inputs is not None:
            if type(inputs) is np.ndarray:
                DATAFRAME = False
                self.training_inputs = inputs
            elif type(inputs) is pd.core.frame.DataFrame:
                DATAFRAME = True
                self.training_inputs = inputs.values
                self.index = list(inputs.index)
            else:
                raise TypeError("Input must be either numpy array or pandas DataFrame!")
        columns_missing = np.where(np.array(np.sum(np.invert(np.isnan(self.training_inputs)), axis=0))==0)[0]
        self.training_inputs = np.delete(self.training_inputs, columns_missing, 1)
        obs = _observations(self.training_inputs)
        glrm_j = j.GLRM(self.training_inputs, self.losses, self.rx, self.ry, self.k, obs=obs, offset=self.offset, scale=self.scale)
        X, Y, ch = j.fit_b(glrm_j)
        self.X = X
        self.Y = Y
        self.fitted = True
        return np.dot(np.transpose(X), Y)

    #store training matrix as attribute of the class
    def set_training_data(self, inputs):
            self.training_inputs = inputs
    
    def produce(self, inputs): #calculate the output map; requires the input to be a numpy array or a pandas DataFrame
        
        if type(inputs) is pd.core.frame.DataFrame:
            inputs = inputs.values
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        #make sure dimension_reduce has already been executed beforehand, and Y and v match in terms of numbers of columns
        try:
            self.Y
        except NameError:
            raise Exception('Initial GLRM fitting not executed!')
        else:
            try:
                if self.Y.shape[1] != inputs.shape[1]:
                    raise ValueError
            except ValueError:
                raise Exception('Dimension of input vector does not match Y!')
            else:
                self.Y = self.Y.astype(float) #make sure column vectors finally have the datatype Array{float64,1} in Julia
                num_cols = self.Y.shape[1]
                ry = [j.FixedLatentFeaturesConstraint(self.Y[:, i]) for i in range(num_cols)]
                glrm_new_j = j.GLRM(inputs, self.losses, self.rx, ry, self.k, offset=self.offset, scale=self.scale)
                x, yp, ch = j.fit_b(glrm_new_j)
                return x


class pca(glrm):
    run_status = 0

class nnmf(glrm):
    def __init__(self, losses = QuadLoss(), rx = NonNegConstraint(), ry = NonNegConstraint(), n_components=2):
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.k = n_components
        self.fitted = False
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}


class rpca(glrm):
    def __init__(self, losses = HuberLoss(), rx = QuadReg(), ry = QuadReg(), n_components=2):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.k = n_components
        self.fitted = False
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry}


# get a list of tuples of observed entries
def observations(inputs, missing_type=np.nan):
    obs = []
    for row in range(inputs.shape[0]):
        for col in range(inputs.shape[1]):
            if np.isnan(missing_type):
                if not np.isnan(inputs)[row, col]:
                    obs.append((row, col))
            else:
                if inputs[row, col] == missing_type:
                    obs.append((row, col))
    return obs

#get a list of tuples of observed entries for Julia
def _observations(inputs, missing_type=np.nan):
    obs = observations(inputs, missing_type=missing_type)
    return [(item[0] + 1, item[1] + 1) for item in obs]




