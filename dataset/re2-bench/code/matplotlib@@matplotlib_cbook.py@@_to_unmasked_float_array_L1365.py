import numpy as np

def _to_unmasked_float_array(x):
    """
    Convert a sequence to a float array; if input was a masked array, masked
    values are converted to nans.
    """
    if hasattr(x, 'mask'):
        return np.ma.asanyarray(x, float).filled(np.nan)
    else:
        return np.asanyarray(x, float)
