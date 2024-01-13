"""Functions used in binary segmentation."""

import numpy as np
import pandas as pd
import pkg_resources



def calculate_stats(series, stat, sigma):
    """Compute statistic along the given series."""

    n = len(series)

    if stat == 'Z':
        cumsum = np.cumsum(series)
        cumsum_rev = np.cumsum(series[::-1])[::-1]
        idx = np.arange(1, n)
        stats = np.abs(cumsum[:-1] / idx - cumsum_rev[1:] / (n - idx)) / (sigma * np.sqrt(1/idx + 1/(n - idx)))

    elif stat == 'T':
        s1 = pd.Series(series).expanding().std().fillna(0).values
        s2 = pd.Series(series[::-1]).expanding().std().fillna(0).values[::-1]
        idx = np.arange(1, n)
        sp = np.sqrt(((idx - 1) * s1[:-1]**2 + (idx[::-1] - 1) * s2[:-1]**2) / (n - 2))

        cumsum = np.cumsum(series)
        cumsum_rev = np.cumsum(series[::-1])[::-1]
        stats = np.abs(cumsum[:-1] / idx - cumsum_rev[1:] / (n - idx)) / (sp * np.sqrt(1/idx + 1/(n - idx)))

    return stats



def quantile(prob, param, stat):
    """Retrieve a quantile of a specified order (`prob`) from the csv files."""

    path = 'quantiles/binseg_' + stat + '.csv'
    stream = pkg_resources.resource_stream(__name__, path)
    df = pd.read_csv(stream)
    q = df['n' + str(param)][np.floor(prob * 1000)]    
    return q



def bootstrap(series, B):
    """Compute B values of the test statistic from B samples generated with bootstrap."""

    n = len(series)
    Ys = np.random.choice(series, size=(B, n), replace=True)

    cumsum = np.cumsum(Ys, axis=1)
    cumsum_rev = np.cumsum(Ys[:, ::-1], axis=1)[:, ::-1]
    idx = np.arange(1, n)
    Zj = np.abs(cumsum[:, :-1] / idx - cumsum_rev[:, 1:] / (n - idx)) / np.sqrt(1/idx + 1/(n - idx))
    stat_max = np.max(Zj, axis=1)

    return stat_max