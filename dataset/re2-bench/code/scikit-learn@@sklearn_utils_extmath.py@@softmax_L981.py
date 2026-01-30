import numpy as np
from sklearn.utils._array_api import (
    _average,
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _nanmean,
    _nansum,
    device,
    get_namespace,
    get_namespace_and_device,
)

def softmax(X, copy=True):
    """
    Calculate the softmax function.

    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X : array-like of float of shape (M, N)
        Argument to the logistic function.

    copy : bool, default=True
        Copy X or not.

    Returns
    -------
    out : ndarray of shape (M, N)
        Softmax function evaluated at every point in x.
    """
    xp, is_array_api_compliant = get_namespace(X)
    if copy:
        X = xp.asarray(X, copy=True)
    max_prob = xp.reshape(xp.max(X, axis=1), (-1, 1))
    X -= max_prob

    if _is_numpy_namespace(xp):
        # optimization for NumPy arrays
        np.exp(X, out=np.asarray(X))
    else:
        # array_api does not have `out=`
        X = xp.exp(X)

    sum_prob = xp.reshape(xp.sum(X, axis=1), (-1, 1))
    X /= sum_prob
    return X
