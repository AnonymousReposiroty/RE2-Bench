def cached_unique(*ys, xp=None):
    """Return the unique values of ys.

    Use the cached values from dtype.metadata if present.

    This function does NOT cache the values in y, i.e. it doesn't change y.

    Call `attach_unique` to attach the unique values to y.

    Parameters
    ----------
    *ys : sequence of array-like
        Input data arrays.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    res : tuple of array-like or array-like
        Unique values of ys.
    """
    res = tuple(_cached_unique(y, xp=xp) for y in ys)
    if len(res) == 1:
        return res[0]
    return res
