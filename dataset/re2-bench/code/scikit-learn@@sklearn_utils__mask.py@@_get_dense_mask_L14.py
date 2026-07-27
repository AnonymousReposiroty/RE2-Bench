from contextlib import suppress
import numpy as np
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils.fixes import _object_dtype_isnan
import pandas

def _get_dense_mask(X, value_to_mask):
    with suppress(ImportError, AttributeError):
        # We also suppress `AttributeError` because older versions of pandas do
        # not have `NA`.
        import pandas

        if value_to_mask is pandas.NA:
            return pandas.isna(X)

    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            Xt = np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            Xt = _object_dtype_isnan(X)
    else:
        Xt = X == value_to_mask

    return Xt
