from sklearn.utils._array_api import get_namespace

def _cached_unique(y, xp=None):
    """Return the unique values of y.

    Use the cached values from dtype.metadata if present.

    This function does NOT cache the values in y, i.e. it doesn't change y.

    Call `attach_unique` to attach the unique values to y.
    """
    try:
        if y.dtype.metadata is not None and "unique" in y.dtype.metadata:
            return y.dtype.metadata["unique"]
    except AttributeError:
        # in case y is not a numpy array
        pass
    xp, _ = get_namespace(y, xp=xp)
    return xp.unique_values(y)
