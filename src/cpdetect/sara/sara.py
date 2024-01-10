

import numpy as np
import pandas as pd

from .utils import bootstrap, quantile



class SaRa():
    """Screening and Ranking algorithm"""

    def __init__(self):
        return
    
    def fit(self, series, stat='Z', sigma=1):
        self.series = series
        self.size = len(series)
        self.stat = stat
        self.sigma = sigma
        return self

    def predict(self, h, alpha, bootstrap_samples=None):
        n = self.size
        statistic = self.stat
        sigma = self.sigma

        if h > n:
            raise ValueError("Parameter h can't be greater than series length.")

        if bootstrap_samples:
            sample = bootstrap(self.series, bootstrap_samples, h)
            q = np.percentile(sample, 100 * (1 - alpha))
            statistic = 'Z'
            sigma = 1
        else:
            q = quantile(1 - alpha, h, self.stat)
        
        if statistic == 'Z':
            S = pd.Series(self.series).rolling(h).sum().dropna()
            Ds = np.abs((S.shift(-h) - S) / h).dropna().values
            Ds = np.hstack((np.zeros(h-1), Ds/sigma, np.zeros(h-1)))

            Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().values
            local_max = np.logical_and(Ds_max == Ds[h-1:n-h], Ds_max > q)

            self.stat_values = Ds
            self.change_points = np.arange(n - 2*h + 1)[local_max] + h
        
        elif statistic == 'T':
            s1 = pd.Series(self.series).rolling(h).std().dropna()
            s2 = s1.shift(-h)
            s_p = np.sqrt((s1**2 + s2**2)/2).dropna().values

            S = pd.Series(self.series).rolling(h).sum().dropna()
            Ds = np.abs((S.shift(-h) - S) / h).dropna().values
            Ds = np.hstack((np.zeros(h-1), Ds/s_p, np.zeros(h-1)))

            Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().values
            local_max = np.logical_and(Ds_max == Ds[h-1:n-h], Ds_max > q)

            self.stat_values = Ds
            self.change_points = np.arange(n - 2*h + 1)[local_max] + h

        return self.change_points