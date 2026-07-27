import numbers
from sklearn.utils._missing import is_pandas_na, is_scalar_nan

def _check_inputs_dtype(X, missing_values):
    if is_pandas_na(missing_values):
        # Allow using `pd.NA` as missing values to impute numerical arrays.
        return
    if X.dtype.kind in ("f", "i", "u") and not isinstance(missing_values, numbers.Real):
        raise ValueError(
            "'X' and 'missing_values' types are expected to be"
            " both numerical. Got X.dtype={} and "
            " type(missing_values)={}.".format(X.dtype, type(missing_values))
        )
