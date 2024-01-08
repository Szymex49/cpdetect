

import numpy as np
import scipy.stats as sp


from .utils import dissim_idx, pooled_std




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
            DI = lambda G1, G2: sp.norm.cdf(dissim_idx(G1, G2) / self.sigma, 0, 1)
            Gs = [ [self.series[0]] ]
            for j in range(1, len(self.series)):
                Gs.append([self.series[j]])
                DIs.append(DI(Gs[j-1], Gs[j]))

        elif self.stat == 'T':
            DI = lambda G1, G2: sp.t.cdf(dissim_idx(G1, G2) / pooled_std(G1, G2), len(G1) + len(G2) - 2)
            Gs = [list(self.series[:2])]
            for j in range(1, int(len(self.series)/2) - 1):
                Gs.append(list(self.series[2*j:2*j+2]))
                DIs.append(DI(Gs[j-1], Gs[j]))
            # Add last segment of size 2 or 3
            last_j = int(len(self.series)/2) - 1 
            Gs.append(list(self.series[2*last_j:]))
            DIs.append(DI(Gs[last_j-1], Gs[last_j]))
        
        while len(Gs) > 2:
            
            n = len(Gs)
            j_min = np.argmin(DIs)
            df = len(Gs[j_min]) + len(Gs[j_min + 1]) - 2

            # Test for equal mean
            if self.stat == 'Z' and min(DIs) > 1 - alpha/2:
                break
            elif self.stat == 'T' and min(DIs) > 1 - alpha/2:
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
                DI_left = DI(Gs[j_min - 1], Gs[j_min])
                DIs.insert(j_min - 1, DI_left)
            
            if j_min < n - 2:
                DI_right = DI(Gs[j_min], Gs[j_min + 1])
                DIs.insert(j_min, DI_right)
        
        if len(Gs) == 2 and DIs[0] < 1 - alpha/2:
            self.change_points = np.array([])
        else:
            self.change_points = np.cumsum(list(map(len, Gs)))[:-1]

        return self.change_points

