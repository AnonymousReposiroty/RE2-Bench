from numbers import Integral, Number
import numpy as np
from matplotlib import _api, _docstring, cbook

def _stride_windows(x, n, noverlap=0):
    _api.check_isinstance(Integral, n=n, noverlap=noverlap)
    x = np.asarray(x)
    step = n - noverlap
    shape = (n, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
