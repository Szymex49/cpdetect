
import numpy as np
import pandas as pd



def pooled_std(X1, X2):
    SD1 = std(X1)
    SD2 = std(X2)
    n1 = np.size(X1)
    n2 = np.size(X2)
    return np.sqrt(((n1 - 1) * SD1**2 + (n2 - 1) * SD2**2) / (n1 + n2 - 2))


