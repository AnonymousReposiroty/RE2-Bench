import numpy as np

def _check_1d(x):
    """Convert scalars to 1D arrays; pass-through arrays as is."""
    # Unpack in case of e.g. Pandas or xarray object
    x = _unpack_to_numpy(x)
    # plot requires `shape` and `ndim`.  If passed an
    # object that doesn't provide them, then force to numpy array.
    # Note this will strip unit information.
    if (not hasattr(x, 'shape') or
            not hasattr(x, 'ndim') or
            len(x.shape) < 1):
        return np.atleast_1d(x)
    else:
        return x
