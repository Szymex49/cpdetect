"""Functions used in Screening and Ranking algorithm."""

import numpy as np
import pandas as pd
import pkg_resources



def quantile(prob, param, stat):
    """Retrieve a quantile of a specified order (`prob`) from the csv files."""

    path = 'quantiles/sara_' + stat + '.csv'
    stream = pkg_resources.resource_stream(__name__, path)
    df = pd.read_csv(stream)
    q = df['n' + str(param)][np.floor(prob * 1000)]    
    return q



def bootstrap(series, B, param=None):
    """Compute B values of the test statistic from B samples generated with bootstrap."""
    
    h = param
    sample = []
    while len(sample) < B:
        Y = np.random.choice(series, size=100000, replace=True)
        S = pd.Series(Y).rolling(h).sum().dropna()
        Ds = np.abs((S.shift(-h) - S) / h).dropna().values
        Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().values
        local_max = Ds_max == Ds[h-1:len(Ds)-h+1]
        sample.extend(Ds_max[local_max]) 
    return sample
