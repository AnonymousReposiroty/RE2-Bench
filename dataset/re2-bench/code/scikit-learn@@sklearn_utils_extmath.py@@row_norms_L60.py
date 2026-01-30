import numpy as np
from scipy import linalg, sparse
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
from sklearn.utils.sparsefuncs_fast import csr_row_norms

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.

    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        X = X.tocsr()
        norms = csr_row_norms(X)
        if not squared:
            norms = np.sqrt(norms)
    else:
        xp, _ = get_namespace(X)
        if _is_numpy_namespace(xp):
            X = np.asarray(X)
            norms = np.einsum("ij,ij->i", X, X)
            norms = xp.asarray(norms)
        else:
            norms = xp.sum(xp.multiply(X, X), axis=1)
        if not squared:
            norms = xp.sqrt(norms)
    return norms
