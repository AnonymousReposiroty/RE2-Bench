import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
from sklearn.externals._packaging.version import parse as parse_version
import scipy  # noqa: F401

def _mode(a, axis=0):
    mode = scipy.stats.mode(a, axis=axis, keepdims=True)
    if sp_version >= parse_version("1.10.999"):
        # scipy.stats.mode has changed returned array shape with axis=None
        # and keepdims=True, see https://github.com/scipy/scipy/pull/17561
        if axis is None:
            mode = np.ravel(mode)
    return mode
