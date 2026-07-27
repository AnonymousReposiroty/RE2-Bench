from scipy.sparse import csr_matrix, issparse
from sklearn.utils._array_api import (
    _fill_diagonal,
    _find_matching_floating_dtype,
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _modify_in_place_if_numpy,
    get_namespace,
    get_namespace_and_device,
)

def _find_floating_dtype_allow_sparse(X, Y, xp=None):
    """Find matching floating type, allowing for sparse input."""
    if any([issparse(X), issparse(Y)]) or _is_numpy_namespace(xp):
        X, Y, dtype_float = _return_float_dtype(X, Y)
    else:
        dtype_float = _find_matching_floating_dtype(X, Y, xp=xp)
    return X, Y, dtype_float
