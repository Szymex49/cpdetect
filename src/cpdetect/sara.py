

import numpy as np
import pandas as pd



def bootstrap(series, B, h):
    sample = []
    while len(sample) < B:
        Y = np.random.choice(series, size=100000, replace=True)
        S = pd.Series(Y).rolling(h).sum().dropna()
        Ds = np.abs((S.shift(-h) - S) / h).dropna().values
        Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().reset_index(drop=True).values
        local_max = Ds_max == Ds[h-1:len(Ds)-h+1]
        sample.extend(Ds_max[local_max]) 
    return sample


class SaRa():
    """Screening and Ranking algorithm"""

    def __init__(self, model='normal mean'):
        if model not in ['normal mean']:
            raise ValueError('No such model.')
        self.model = model
    
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

        if bootstrap_samples:
            sample = bootstrap(self.series, bootstrap_samples, h)
            q = np.percentile(sample, 100 * (1 - alpha))
            statistic = 'Z'
            sigma = 1
        else:
            q = quantile(1 - alpha, h, 'sara', self.stat)
        
        if statistic == 'Z':
            S = pd.Series(self.series).rolling(h).sum().dropna()
            Ds = np.abs((S.shift(-h) - S) / h).dropna().values
            Ds = np.hstack((np.zeros(h-1), Ds/sigma, np.zeros(h-1)))

            Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().reset_index(drop=True).values
            local_max = np.logical_and(Ds_max == Ds[h-1:n-h], Ds_max > q)

            self.stat_values = Ds
            self.change_points = np.arange(n - 2*h + 1)[local_max] + h
        
        elif statistic == 'T':
            s1 = pd.Series(self.series).rolling(h).std().dropna()
            s2 = s1.shift(-h)
            sp = np.sqrt((s1**2 + s2**2)/2).dropna().values

            S = pd.Series(self.series).rolling(h).sum().dropna()
            Ds = np.abs((S.shift(-h) - S) / h).dropna().values
            Ds = np.hstack((np.zeros(h-1), Ds/sp, np.zeros(h-1)))

            Ds_max = pd.Series(Ds).rolling(2*h - 1).max().dropna().reset_index(drop=True).values
            local_max = np.logical_and(Ds_max == Ds[h-1:n-h], Ds_max > q)

            self.stat_values = Ds
            self.change_points = np.arange(n - 2*h + 1)[local_max] + h

        return self.change_points