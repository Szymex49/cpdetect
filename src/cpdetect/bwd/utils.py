"""Functions used in backward detection."""

import numpy as np


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
    """Calculate pooled standard deviation."""
    SD1 = std(X1)
    SD2 = std(X2)
    n1 = np.size(X1)
    n2 = np.size(X2)
    return np.sqrt(((n1 - 1) * SD1**2 + (n2 - 1) * SD2**2) / (n1 + n2 - 2))



def dissim_idx(G1, G2):
    """Compute dissimilarity index between G1 and G2."""
    return abs(np.mean(G1) - np.mean(G2)) / np.sqrt(1/len(G1) + 1/len(G2))

