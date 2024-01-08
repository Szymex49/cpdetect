
import numpy as np
import pandas as pd



def var(X, axis=0):
    if np.size(X) == 1:
        return 0
    else:
        return np.var(X, axis=axis) * np.size(X, axis=axis) / (np.size(X, axis=axis) - 1)

def std(X, axis=0):
    if np.size(X) == 1:
        return 0
    else:
        return np.sqrt(var(X, axis=axis))



def pooled_std(X1, X2):
    SD1 = std(X1)
    SD2 = std(X2)
    n1 = np.size(X1)
    n2 = np.size(X2)
    return np.sqrt(((n1 - 1) * SD1**2 + (n2 - 1) * SD2**2) / (n1 + n2 - 2))



def dissim_idx(G1, G2):
    return abs(np.mean(G1) - np.mean(G2)) / np.sqrt(1/len(G1) + 1/len(G2))



def calculate_stats(series, stat, sigma):
    """Calculates statistic on given series."""

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



def quantile(prob, param, method, stat):
    file = method + '_' + stat + '.csv'
    df = pd.read_csv('quantiles/' + file)
    q = df['n' + str(param)][np.floor(prob * 1000)]    
    return q



def bootstrap(series, B, method, param=None):

    if method == 'binseg':
        n = len(series)
        Ys = np.random.choice(series, size=(B, n), replace=True)

        cumsum = np.cumsum(Ys, axis=1)
        cumsum_rev = np.cumsum(Ys[:, ::-1], axis=1)[:, ::-1]
        idx = np.arange(1, n)
        Zj = np.abs(cumsum[:, :-1] / idx - cumsum_rev[:, 1:] / (n - idx)) / np.sqrt(1/idx + 1/(n - idx))
        stat_max = np.max(Zj, axis=1)

        return stat_max
    
    elif method == 'sara':
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

    else:
        raise ValueError('Method does not support bootstrap.')


