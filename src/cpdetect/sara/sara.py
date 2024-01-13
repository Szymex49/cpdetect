"""Screening and Ranking algorithm"""


import numpy as np
import pandas as pd
from typing import Iterable

from .utils import bootstrap, quantile



class SaRa():
    """Screening and Ranking algorithm"""

    def __init__(self):
        return
    

    def fit(
        self,
        series: Iterable,
        stat: str = 'Z',
        sigma: float = 1
    ):
        """Fit the model to the data.

        Args:
            series (Iterable): time series to be analyzed, must be one-dimensional
            stat (str): statistic used for tests on change-point ("Z" or "T")
            sigma (float): theoretical standard deviation of time series values (only required when Z statistic was chosen)

        Returns:
            self
        """

        self.series = series
        self.size = len(series)
        self.stat = stat
        self.sigma = sigma
        return self


    def predict(
        self,
        h: int,
        alpha: float,
        bootstrap_samples: int = None
    ) -> np.ndarray:
        
        """Detect change-points along the series.

        Args:
            h (int): SaRa h parameter where 2h is a window size
            alpha (float): significance level of a change-point test
            bootstrap_samples (int): size of a test statistic sample which is generated with bootstrap, used for computing 1 - alpha quantile of the test statistic distribution

        Returns:
            array: vector of change-points
        """
        
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