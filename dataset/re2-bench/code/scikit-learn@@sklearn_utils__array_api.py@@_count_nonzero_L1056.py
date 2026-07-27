import scipy.sparse as sp
from sklearn.utils.sparsefuncs import count_nonzero

def _count_nonzero(X, axis=None, sample_weight=None, xp=None, device=None):
    """A variant of `sklearn.utils.sparsefuncs.count_nonzero` for the Array API.

    If the array `X` is sparse, and we are using the numpy namespace then we
    simply call the original function. This function only supports 2D arrays.
    """
    from sklearn.utils.sparsefuncs import count_nonzero

    xp, _ = get_namespace(X, sample_weight, xp=xp)
    if _is_numpy_namespace(xp) and sp.issparse(X):
        return count_nonzero(X, axis=axis, sample_weight=sample_weight)

    assert X.ndim == 2

    weights = xp.ones_like(X, device=device)
    if sample_weight is not None:
        sample_weight = xp.asarray(sample_weight, device=device)
        sample_weight = xp.reshape(sample_weight, (sample_weight.shape[0], 1))
        weights = xp.astype(weights, sample_weight.dtype) * sample_weight

    zero_scalar = xp.asarray(0, device=device, dtype=weights.dtype)
    return xp.sum(xp.where(X != 0, weights, zero_scalar), axis=axis)
