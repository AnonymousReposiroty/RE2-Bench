from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot, squared_norm

def sandwich_dot(X, W):
    """Compute the sandwich product X.T @ diag(W) @ X."""
    # TODO: This "sandwich product" is the main computational bottleneck for solvers
    # that use the full hessian matrix. Here, thread parallelism would pay-off the
    # most.
    # While a dedicated Cython routine could exploit the symmetry, it is very hard to
    # beat BLAS GEMM, even thought the latter cannot exploit the symmetry, unless one
    # pays the price of taking square roots and implements
    #    sqrtWX = sqrt(W)[: None] * X
    #    return sqrtWX.T @ sqrtWX
    # which (might) detect the symmetry and use BLAS SYRK under the hood.
    n_samples = X.shape[0]
    if sparse.issparse(X):
        return safe_sparse_dot(
            X.T,
            sparse.dia_matrix((W, 0), shape=(n_samples, n_samples)) @ X,
            dense_output=True,
        )
    else:
        # np.einsum may use less memory but the following, using BLAS matrix
        # multiplication (GEMM), is by far faster.
        WX = W[:, None] * X
        return X.T @ WX
