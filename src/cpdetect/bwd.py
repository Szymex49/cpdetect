



import numpy as np
import scipy.stats as sp

from tools import dissim_idx, pooled_std




class BWD():
    """Backward detection"""

    def __init__(self):
        return
    
    def fit(self, series, stat='Z', sigma=1):
        self.series = series
        self.stat = stat
        self.sigma = sigma
        return self
    
    def predict(self, alpha):
        """Predict"""

        DIs = []

        if self.stat == 'Z':
            Gs = [ [self.series[0]] ]
            for j in range(1, len(self.series)):
                Gs.append([self.series[j]])
                DIs.append(dissim_idx(Gs[j-1], Gs[j]))

            quant = lambda df: sp.norm.ppf(1 - alpha/2, 0, 1)

        elif self.stat == 'T':
            Gs = [list(self.series[:2])]
            for j in range(1, int(len(self.series)/2) - 1):
                Gs.append(list(self.series[2*j:2*j+2]))
                DIs.append(dissim_idx(Gs[j-1], Gs[j]))
            # Add last segment of size 2 or 3
            last_j = int(len(self.series)/2) - 1 
            Gs.append(list(self.series[2*last_j:]))
            DIs.append(dissim_idx(Gs[last_j-1], Gs[last_j]))
            
            quant = lambda df: sp.t.ppf(1 - alpha/2, df)
        
        while len(Gs) > 2:
            
            n = len(Gs)
            j_min = np.argmin(DIs)
            df = len(Gs[j_min]) + len(Gs[j_min + 1]) - 2

            # Test for equal mean
            if self.stat == 'Z' and min(DIs)/self.sigma > quant(df):
                break
            elif self.stat == 'T' and min(DIs)/pooled_std(Gs[j_min], Gs[j_min + 1]) > quant(df):
                break

            G = Gs[j_min] + Gs[j_min + 1]
            Gs.pop(j_min + 1)
            Gs.pop(j_min)
            Gs.insert(j_min, G)

            if j_min < n - 2:
                DIs.pop(j_min + 1)
            DIs.pop(j_min)
            if j_min > 0:
                DIs.pop(j_min - 1)
            
            if j_min > 0:
                DI_left = dissim_idx(Gs[j_min - 1], Gs[j_min])
                DIs.insert(j_min - 1, DI_left)
            
            if j_min < n - 2:
                DI_right = dissim_idx(Gs[j_min], Gs[j_min + 1])
                DIs.insert(j_min, DI_right)
        
        if len(Gs) == 2 and DIs[0] < quant(n - 2):
            self.change_points = np.array([])
        else:
            self.change_points = np.cumsum(list(map(len, Gs)))[:-1]

        return self.change_points

