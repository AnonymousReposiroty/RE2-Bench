from scipy.sparse.linalg import LinearOperator

def _implicit_column_offset(X, offset):
    """Create an implicitly offset linear operator.

    This is used by PCA on sparse data to avoid densifying the whole data
    matrix.

    Params
    ------
        X : sparse matrix of shape (n_samples, n_features)
        offset : ndarray of shape (n_features,)

    Returns
    -------
    centered : LinearOperator
    """
    offset = offset[None, :]
    XT = X.T
    return LinearOperator(
        matvec=lambda x: X @ x - offset @ x,
        matmat=lambda x: X @ x - offset @ x,
        rmatvec=lambda x: XT @ x - (offset * x.sum()),
        rmatmat=lambda x: XT @ x - offset.T @ x.sum(axis=0)[None, :],
        dtype=X.dtype,
        shape=X.shape,
    )
