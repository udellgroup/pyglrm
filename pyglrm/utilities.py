import julia
import numpy as np
import pandas as pd
from typing import Union

j = julia.Julia()
j.using("LowRankModels")

__all__ = ['observations']

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
