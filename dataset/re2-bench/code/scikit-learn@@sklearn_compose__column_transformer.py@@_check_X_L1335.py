from scipy import sparse
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_n_features,
    _get_feature_names,
    _is_pandas_df,
    _num_samples,
    check_array,
    check_is_fitted,
    validate_data,
)

def _check_X(X):
    """Use check_array only when necessary, e.g. on lists and other non-array-likes."""
    if (
        (hasattr(X, "__array__") and hasattr(X, "shape"))
        or hasattr(X, "__dataframe__")
        or sparse.issparse(X)
    ):
        return X
    return check_array(X, ensure_all_finite="allow-nan", dtype=object)
