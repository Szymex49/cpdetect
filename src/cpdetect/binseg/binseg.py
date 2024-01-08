


import numpy as np
import pandas as pd
from typing import Iterable

from .utils import quantile, bootstrap, calculate_stats




class BinSeg():
    """Binary segmentation"""

    def __init__(self):
        return
    
    def fit(self, series:Iterable, stat='Z', sigma=1):
        self.series = series
        self.size = len(series)
        self.stat = stat
        self.sigma = sigma
        self.stat_values = calculate_stats(self.series, stat=stat, sigma=sigma)
        return self

    def predict(self, alpha, bootstrap_samples=None):
        js = []
        intervals = []
        intervals.append(self.series)
        first_indexes = []
        first_indexes.append(1)

        if bootstrap_samples:
            quant = lambda Y: np.percentile(bootstrap(Y, bootstrap_samples), 100 * (1 - alpha))
            calc_stats = lambda Y: calculate_stats(Y, stat='Z', sigma=1)
        else:
            quant = lambda Y: quantile(1 - alpha, len(Y), self.stat)
            calc_stats = lambda Y: calculate_stats(Y, stat=self.stat, sigma=self.sigma)
        
        if self.stat == 'Z':
            min_interval_len = 3
        elif self.stat == 'T':
            min_interval_len = 4

        while len(intervals) > 0:
            Y = intervals[0]
            first_index = first_indexes[0]

            stats = calc_stats(Y)
            stat_max = max(stats)
            j_max = np.argmax(stats) + 1

            q = quant(Y)
            if stat_max > q:
                js.append(first_index + j_max - 1)

                if len(Y[:j_max]) >= min_interval_len:
                    intervals.append(Y[:j_max])
                    first_indexes.append(first_index)

                if len(Y[j_max:]) >= min_interval_len:
                    intervals.append(Y[j_max:])
                    first_indexes.append(first_index + j_max)

            intervals.pop(0)
            first_indexes.pop(0)
        
        self.change_points = sorted(js)
        return self.change_points
