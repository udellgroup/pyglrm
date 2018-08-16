import julia
import numpy as np
import pandas as pd
from typing import Union

j = julia.Julia()
j.using("LowRankModels")

__all__ = ['QuadLoss','L1Loss','HuberLoss', 'HingeLoss', 'WeightedHingeLoss', 'PeriodicLoss', 'MultinomialLoss', 'MultinomialOrdinalLoss', 'ZeroReg','NonNegOneReg','QuadReg', 'QuadConstraint', 'NonNegConstraint', 'OneReg', 'OneSparseConstraint', 'UnitOneSparseConstraint', 'SimplexConstraint', 'FixedLatentFeaturesConstraint', 'glrm', 'pca', 'nnmf', 'rpca', 'observations']

#Losses
# TODO: add PoissonLoss
def QuadLoss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.QuadLoss(scale, domain)

def L1Loss(scale=1.0, domain=j.RealDomain()):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.L1Loss(scale, domain)

def HuberLoss(scale=1.0, domain=j.RealDomain(), crossover=1.0):
    if not isinstance(scale, Union[float, int].__args__) or not isinstance(crossover, Union[float, int].__args__):
        raise TypeError
    return j.HuberLoss(scale, domain, crossover)

def HingeLoss(min=1, max=10, scale=1.0):
    if not isinstance(min, int) or not isinstance(max, int) or not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.OrdinalHingeLoss(min, max, scale, j.OrdinalDomain(min, max))

def WeightedHingeLoss(scale=1.0, case_weight_ratio=1.0):
    if not isinstance(scale, Union[float, int].__args__) or not isinstance(case_weight_ratio, Union[float, int].__args__):
        raise TypeError
    return j.WeightedHingeLoss(scale, j.BoolDomain(), case_weight_ratio)

def PeriodicLoss(T, scale=1.0):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.PeriodicLoss(T, scale)

def MultinomialLoss(max, scale=1.0):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.MultinomialLoss(max, scale)

def MultinomialOrdinalLoss(max, scale=1.0):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.MultinomialOrdinalLoss(max, scale)

#Regularizers
def ZeroReg():
    return j.ZeroReg()

def NonNegOneReg(scale=1):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.NonNegOneReg(scale)

def QuadReg(scale=1):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.QuadReg(scale)

def QuadConstraint(max_2norm=1):
    if not isinstance(max_2norm, Union[float, int].__args__):
        raise TypeError
    return j.QuadConstraint(max_2norm)

def NonNegConstraint():
    return j.NonNegConstraint()

def OneReg(scale=1):
    if not isinstance(scale, Union[float, int].__args__):
        raise TypeError
    return j.OneReg(scale)

def OneSparseConstraint():
    return j.OneSparseConstraint()

def UnitOneSparseConstraint():
    return j.UnitOneSparseConstraint()

def SimplexConstraint():
    return j.SimplexConstraint()

def FixedLatentFeaturesConstraint(Y):
    return j.FixedLatentFeaturesConstraint(Y)


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
        n_components (int):                                 The maximum rank of X and Y, namely, the number of components to keep.
        training_inputs (numpy.ndarray or DataFrame):       The matrix to be decomposed.
        index (list):                                       Indices of the input DataFrame (None when input is numpy.ndarray).
        fitted (bool):                                      Whether this glrm has been fitted at least once.
        offset (bool):                                      Whether to do offsetting before matrix decomposition.
        scale (bool):                                       Whether to do scaling before matrix decomposition.
        obs (list):                                         A list of indices of observed entries (for missing entry imputation).
        hyperparams (dict):                                 A dictionary of losses, regularizers, observed entry indices, and whether to offset or scale.
        
    """
    is_dimensionality_reduction = True
    is_feature_selection = False

    def __init__(self, losses = QuadLoss(), rx = ZeroReg(), ry = ZeroReg(), n_components=2, X=None, Y=None, offset=False, scale=False):
        self.losses = losses
        self.rx = rx
        self.ry = ry
        self.n_components = n_components
        self.training_inputs = None
        self.index = None
        self.fitted = False
        self.offset = offset
        self.scale = scale
        self.obs = None
        self.hyperparams = {'losses': self.losses, 'rx': self.rx, 'ry': self.ry, 'n_components': self.n_components, 'obs': self.obs, 'offset': self.offset, 'scale': self.scale}
        
        # TODO: add the option of initializing with SVD
        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y
    

    def fit(self, inputs=None, identify_obs=True):
        """
        Fit the GLRM model on current inputs (if the inputs argument is not None) or previous (if the inputs argument is None), WITHOUT returning the fitted matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The matrix to be fitted.
        identify_obs (bool):                                             Whether to automatically get indices of missing entries.
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
        if identify_obs:
            self.obs = observations(self.training_inputs)
        # TODO: automatically choose default loss and regularizer types by column data types
        glrm_j = j.GLRM(self.training_inputs, self.losses, self.rx, self.ry, self.n_components, obs=_convert_observations(self.obs), offset=self.offset, scale=self.scale)
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

    def fit_impute(self, inputs=None, identify_obs=True):
        """
        In an unsupervised way, fit GLRM onto current input (if the inputs argument is not None) or previous one (if the inputs argument is None), AND return the dimensionality-reduced matrix with the same shape as the original matrix.
        
        Args:
        inputs (np.ndarray or DataFrame, shape (n_samples, n_features)): The original matrix to fit GLRM onto.
        identify_obs (bool):                                             Whether to automatically get indices of missing entries.
        
        Returns:
        (np.ndarray or DataFrame), shape (n_samples, n_features)): The dimensionality-reduced matrix.
        """
        self.fit(inputs=inputs, identify_obs=identify_obs)
        if not self.DATAFRAME:
            return np.dot(self.X.T, self.Y)
        else:
            return pd.DataFrame(np.dot(self.X.T, self.Y), index=self.index)


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

    def get_params(self):
        """
        Returns the parameters of GLRM, i.e., the Y matrix representing generalized principal components.
        
        Returns:
        Y (np.ndarray or DataFrame, shape (n_components, n_features)): The Y matrix to be used in GLRM.
        """
        return self.Y
    
    def set_hyperparams(self, losses=None, rx=None, ry=None, n_components=None, obs=None, offset=None, scale=None):
        """
        Set the hyperparameters of GLRM, i.e., a dictionary of losses, regularizers, observed entry indices, and whether to offset or scale.
        
        Args:
        losses:                                             An individual or a list of loss type.
        rx:                                                 An individual or a list of regularizer on X.
        ry:                                                 An individual or a list of regularizer on Y.
        n_components (int):                                 The maximum rank of X and Y, namely, the number of components to keep.
        obs (list):                                         A list of indices of observed entries (for missing entry imputation).
        offset (bool):                                      Whether to do offsetting before matrix decomposition.
        scale (bool):                                       Whether to do scaling before matrix decomposition.
        """
        for key in list(self.hyperparams.keys()):
            if eval(key) is not None:
                setattr(self, key, eval(key))
    
    def get_hyperprams(self):
        """
        Get the hyperparameters of GLRM, i.e., a dictionary of losses, regularizers, observed entry indices, and whether to offset or scale.
        
        Returns:
        hyperparams (dict): A dictionary of losses, regularizers, observed entry indices, and whether to offset or scale.
        """
        return {key: getattr(self, key) for key in self.hyperparams.keys()}
    
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
                glrm_new_j = j.GLRM(inputs, self.losses, self.rx, ry, self.n_components, offset=self.offset, scale=self.scale)
                x, yp, ch = j.fit_b(glrm_new_j)
                return x


class pca(glrm):
    """
        The class for principal component analysis (PCA).
    """
    pass

class nnmf(glrm):
    """
        The class for nonnegative matrix factorization (NNMF).
    """
    def __init__(self, n_components=2, X=None, Y=None, offset=False, scale=False):
        super().__init__(n_components=n_components, X=X, Y=Y, offset=offset, scale=scale)
        self.losses = QuadLoss()
        self.rx = NonNegConstraint()
        self.ry = NonNegConstraint()


class rpca(glrm):
    """
        The class for robust PCA.
    """
    def __init__(self, n_components=2, X=None, Y=None, offset=False, scale=False):
        super().__init__(n_components=n_components, X=X, Y=Y, offset=offset, scale=scale)
        self.losses = HuberLoss()
        self.rx = QuadReg()
        self.ry = QuadReg()


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
                if inputs[row, col] != missing_type:
                    obs.append((row, col))
    return obs

# below are private functions
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

def _convert_observations(obs, python_to_julia=True):
    """
    For internal use only; convert the list of Python indices to Julia, or reversely.
    
    Args:
    obs (list): A list of observed indices.
    
    Returns:
    obs (list): A list of observed indices in another language.
    """
    if python_to_julia:
        return [(item[0] + 1, item[1] + 1) for item in obs]
    else:
        return [(item[0] - 1, item[1] - 1) for item in obs]




