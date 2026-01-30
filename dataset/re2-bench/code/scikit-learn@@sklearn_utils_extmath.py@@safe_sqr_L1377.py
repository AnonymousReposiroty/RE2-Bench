from scipy import linalg, sparse
from sklearn.utils.validation import check_array, check_random_state

def safe_sqr(X, *, copy=True):
    """Element wise squaring of array-likes and sparse matrices.

    Parameters
    ----------
    X : {array-like, ndarray, sparse matrix}

    copy : bool, default=True
        Whether to create a copy of X and operate on it or to perform
        inplace computation (default behaviour).

    Returns
    -------
    X ** 2 : element wise square
         Return the element-wise square of the input.

    Examples
    --------
    >>> from sklearn.utils import safe_sqr
    >>> safe_sqr([1, 2, 3])
    array([1, 4, 9])
    """
    X = check_array(X, accept_sparse=["csr", "csc", "coo"], ensure_2d=False)
    if sparse.issparse(X):
        if copy:
            X = X.copy()
        X.data **= 2
    else:
        if copy:
            X = X**2
        else:
            X **= 2
    return X
