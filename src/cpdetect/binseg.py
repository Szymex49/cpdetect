


import numpy as np
import pandas as pd




def calculate_stats(series, stat, sigma):
    """Calculates statistic on given series."""

    n = len(series)

    if stat == 'Z':
        if sigma == None:
            raise ValueError('Parameter sigma is necessary to calculate Z.')
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


def quantile(prob, param, method, stat):
    file = method + '_' + stat + '.csv'
    df = pd.read_csv('data/' + file)
    q = df['n' + str(param)][np.floor(prob * 1000)]    
    return q


def bootstrap(series, B):
    n = len(series)
    Ys = np.random.choice(series, size=(B, n), replace=True)

    cumsum = np.cumsum(Ys, axis=1)
    cumsum_rev = np.cumsum(Ys[:, ::-1], axis=1)[:, ::-1]
    idx = np.arange(1, n)
    Zj = np.abs(cumsum[:, :-1] / idx - cumsum_rev[:, 1:] / (n - idx)) / np.sqrt(1/idx + 1/(n - idx))
    stat_max = np.max(Zj, axis=1)

    return stat_max



class BinSeg():
    """Binary segmentation"""

    def __init__(self, model='normal mean'):
        if model not in ['normal mean']:
            raise ValueError('No such model.')
        self.model = model
    
    def fit(self, series, stat='Z', sigma=1):
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
            quant = lambda Y: quantile(1 - alpha, len(Y), 'binseg', self.stat)
            calc_stats = lambda Y: calculate_stats(Y, stat=self.stat, sigma=self.sigma)

        while len(intervals) > 0:
            Y = intervals[0]
            first_index = first_indexes[0]

            stats = calc_stats(Y)
            stat_max = max(stats)
            j_ = np.argmax(stats) + 1

            q = quant(Y)
            if stat_max > q:
                js.append(first_index + j_)

                if len(Y[:j_]) > 2:
                    intervals.append(Y[:j_])
                    first_indexes.append(first_index)

                if len(Y[j_:]) > 2:
                    intervals.append(Y[j_:])
                    first_indexes.append(first_index + j_)

            intervals.pop(0)
            first_indexes.pop(0)

        self.change_points = sorted(js)
        return self.change_points