import julia
import numpy as np
import pandas as pd
from typing import Union

j = julia.Julia()
j.using("LowRankModels")

__all__ = ['QuadLoss','L1Loss','HuberLoss', 'HingeLoss', 'WeightedHingeLoss', 'PeriodicLoss', 'MultinomialLoss', 'MultinomialOrdinalLoss']

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
