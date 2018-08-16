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

class glrm:
    """
        The class for generalized low rank models.
        
        Attributes:
        is_dimensionality_reduction (bool):                 Whether to do dimensionality reduction.
        is_feature_selection (bool):                        Whether to do feature selection.
        losses:                                             An individual or a list of loss type.
        rx:                                                 An individual or a list of regularizer on X.
        ry:                                                 An individual or a list of regularizer on Y.
        X (numpy.ndarray, shape(n_components, n_samples)):  The dimensionality-reduced data points.
        Y (numpy.ndarray, shape(n_components, n_features)): (Generalized) principal components.
        k (int):                                            The maximum rank of X and Y.
        training_inputs (numpy.ndarray or DataFrame):       The matrix to be decomposed.
        index (list):                                       Indices of the input DataFrame (None when input is numpy.ndarray).
        fitted (bool):                                      Whether this glrm has been fitted at least once.
        offset (bool):                                      Whether to do offsetting before matrix decomposition.
        scale (bool):                                       Whether to do scaling before matrix decomposition.
        hyperparameters (dict):                             A dictionary of losses, regularizers, offset and scale.
        
    """
    is_dimensionality_reduction = True
    is_feature_selection = False

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
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry, 'offset':offset, 'scale':scale}
        
        # TODO: add the option of initializing with SVD
        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y
    

    def fit(self, inputs=None):
        """
        Fit the GLRM model on current inputs (if the inputs argument is not None) or previous (if the inputs argument is None), WITHOUT returning the fitted matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The matrix to be fitted.
        """
        #no need to fit again if the GLRM has already been fitted and no new inputs are given
        if self.fitted and inputs is None:
            return
        #check if matrix to be fitted is missing
        if inputs is None and self.training_inputs is None:
            raise ValueError("Missing training data.")
        elif inputs is not None:
            assert type(inputs) is np.ndarray or pd.core.frame.DataFrame, "Input type must be either numpy array or pandas DataFrame!"
            if type(inputs) is np.ndarray:
                self.DATAFRAME = False
                self.training_inputs = inputs
            else:
                self.DATAFRAME = True
                self.training_inputs = inputs.values
                self.index = list(inputs.index)
        #eliminate the column with only missing entries
        columns_missing = np.where(np.array(np.sum(np.invert(np.isnan(self.training_inputs)), axis=0))==0)[0]
        self.training_inputs = np.delete(self.training_inputs, columns_missing, 1)
        obs = _observations(self.training_inputs)
        glrm_j = j.GLRM(self.training_inputs, self.losses, self.rx, self.ry, self.k, obs=obs, offset=self.offset, scale=self.scale)
        X, Y, ch = j.fit_b(glrm_j)
        self.X = X
        self.Y = Y
        self.fitted = True

    def fit_transform(self, inputs=None):
        """
        In an unsupervised way, fit GLRM onto current input (if the inputs argument is not None) or previous one (if the inputs argument is None), AND return the dimensionality-reduced matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The original matrix to fit GLRM onto.
        
        Returns:
        (np.ndarray or DataFrame), shape (n_samples, n_components)): The dimensionality-reduced matrix.
        """
        self.fit(inputs=inputs)
        if not self.DATAFRAME:
            return self.X.T
        else:
            return pd.DataFrame(self.X.T, index=self.index)


    def set_training_data(self, inputs):
        """
        Set the matrix to be factorized.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The matrix to be fitted.
        """
        if type(inputs) is np.ndarray:
            self.DATAFRAME = False
            self.training_inputs = inputs
        else:
            self.DATAFRAME = True
            self.training_inputs = inputs.values
            self.index = list(inputs.index)
                    
    def set_params(self, Y):
        """
        Set the parameters of GLRM, i.e., the Y matrix representing generalized principal components.
        
        Args:
        Y (np.ndarray or DataFrame, shape (n_components, n_features)): The Y matrix to be used in GLRM.
        """
        if type(Y) is np.ndarray:
            self.Y = Y
        else:
            self.Y = Y.values
    
    def transform(self, inputs):
        """
        In a supervised way, apply the already fitted GLRM onto the new input matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): New data.
        
        Returns:
        x (np.ndarray or DataFrame, shape (n_samples, n_components)): Dimensionality-reduced data.
        """
        if type(inputs) is pd.core.frame.DataFrame:
            inputs = inputs.values
            index = list(inputs.index)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
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
    """
        The class for principal component analysis (PCA).
    """
    run_status = 0

class nnmf(glrm):
    """
        The class for nonnegative matrix factorization (NNMF).
    """
    def __init__(self, losses = QuadLoss(), rx = NonNegConstraint(), ry = NonNegConstraint(), n_components=2):
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.k = n_components
        self.fitted = False
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry, 'offset':offset, 'scale':scale}

class rpca(glrm):
    """
        The class for robust PCA.
    """
    def __init__(self, losses = HuberLoss(), rx = QuadReg(), ry = QuadReg(), n_components=2):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.k = n_components
        self.fitted = False
        self.hyperparameters = {'losses':losses, 'rx':rx, 'ry':ry, 'offset':offset, 'scale':scale}


def observations(inputs, missing_type=np.nan):
    """
        Get the positions of observed entries in the input matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The data.
        
        Returns:
        obs (list): A list of tuples representing positions of observed entries.
    """
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
    """
        For internal use only; get the positions of observed entries in the input matrix and pass into Julia.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The data.
        
        Returns:
        obs (list): A list of tuples representing positions of observed entries; indices start from 1.
    """
    obs = observations(inputs, missing_type=missing_type)
    return [(item[0] + 1, item[1] + 1) for item in obs]




