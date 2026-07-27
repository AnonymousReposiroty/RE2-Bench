import numpy as np
from scipy.sparse import dia_matrix, issparse
from sklearn.utils.extmath import _randomized_svd, make_nonnegative, safe_sparse_dot

def _log_normalize(X):
    """Normalize ``X`` according to Kluger's log-interactions scheme."""
    X = make_nonnegative(X, min_value=1)
    if issparse(X):
        raise ValueError(
            "Cannot compute log of a sparse matrix,"
            " because log(x) diverges to -infinity as x"
            " goes to 0."
        )
    L = np.log(X)
    row_avg = L.mean(axis=1)[:, np.newaxis]
    col_avg = L.mean(axis=0)
    avg = L.mean()
    return L - row_avg - col_avg + avg
