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

def _nanaverage(a, weights=None):
    """Compute the weighted average, ignoring NaNs.

    Parameters
    ----------
    a : ndarray
        Array containing data to be averaged.
    weights : array-like, default=None
        An array of weights associated with the values in a. Each value in a
        contributes to the average according to its associated weight. The
        weights array can either be 1-D of the same shape as a. If `weights=None`,
        then all data in a are assumed to have a weight equal to one.

    Returns
    -------
    weighted_average : float
        The weighted average.

    Notes
    -----
    This wrapper to combine :func:`numpy.average` and :func:`numpy.nanmean`, so
    that :func:`np.nan` values are ignored from the average and weights can
    be passed. Note that when possible, we delegate to the prime methods.
    """
    xp, _ = get_namespace(a)
    if a.shape[0] == 0:
        return xp.nan

    mask = xp.isnan(a)
    if xp.all(mask):
        return xp.nan

    if weights is None:
        return _nanmean(a, xp=xp)

    weights = xp.asarray(weights)
    a, weights = a[~mask], weights[~mask]
    try:
        return _average(a, weights=weights)
    except ZeroDivisionError:
        # this is when all weights are zero, then ignore them
        return _average(a)
