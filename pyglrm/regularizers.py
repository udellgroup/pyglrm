import julia
import numpy as np
import pandas as pd
from typing import Union

j = julia.Julia()
j.using("LowRankModels")

__all__ = ['ZeroReg','NonNegOneReg','QuadReg', 'QuadConstraint', 'NonNegConstraint', 'OneReg', 'OneSparseConstraint', 'UnitOneSparseConstraint', 'SimplexConstraint', 'FixedLatentFeaturesConstraint']


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
