# Change-point detection with cpdetect

[![PyPI version](https://badge.fury.io/py/cpdetect.svg)](https://badge.fury.io/py/cpdetect)
![GitHub Release Date](https://img.shields.io/github/release-date/Szymex49/cpdetect)
[![Downloads](https://static.pepy.tech/badge/cpdetect)](https://pepy.tech/project/cpdetect)


**cpdetect** is a python package designed for change point-detection using statistical methods. This is a first version and considers only one change-point model which is normal mean model. This assumes normally distributed time series values and changes only in the mean (mean shift). The package offers three detection methods which are Binary Segmentation (BS), Backward Detection (BWD) and Screening and Ranking algorithm (SaRa).


## Install

To install the package use

    pip install cpdetect


## Example usage

Let's import some useful libraries first.
```python
import scipy.stats as sp
import numpy as np
from matplotlib import pyplot as plt
```

Now we can create an example time series with three change-points.
```python
# mean shifts between 0 and 3, sigma = 2
Y1 = sp.norm.rvs(0, 2, 200)
Y2 = sp.norm.rvs(3, 2, 200)
Y3 = sp.norm.rvs(0, 2, 200)
Y4 = sp.norm.rvs(3, 2, 200)
Y = np.hstack((Y1, Y2, Y3, Y4))
plt.plot(Y)
```
<img src="./images/mean_shift_example.jpg" width="500">

To find the change-points location we can use `BinSeg` which contains binary segmentation implementation.
```python
from cpdetect import BinSeg

bs = BinSeg()                 # creating object

bs.fit(Y, stat='Z', sigma=2)  # fitting to data

plt.plot(bs.stat_values)      # statistic plot

bs.predict(0.01)              # change-point detection
```
<img src="./images/bs_Z_plot.jpg" width="500">

If we don't know what the standard deviation (`sigma`) is, we can use T statistic.
```python
bs.fit(Y, stat='T')        # fitting to data

plt.plot(bs.stat_values)   # statistic plot

bs.predict(0.01)           # change-point detection
```
<img src="./images/bs_T_plot.jpg" width="500">

If we don't know the distribution of time series values, we can't use normal mean models. Then we can use bootstrap which finds the statistic distribution by itself.
```python
bs.predict(0.01, bootstrap_samples=1000)
```



## Libraries used
- `numpy`
- `pandas`
- `scipy`
