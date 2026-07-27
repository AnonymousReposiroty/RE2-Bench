import numpy as np
from scipy import linalg, optimize, sparse
from sklearn.utils.extmath import row_norms, safe_sparse_dot

def _solve_cholesky(X, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_features = X.shape[1]
    n_targets = y.shape[1]

    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)

    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    if one_alpha:
        A.flat[:: n_features + 1] += alpha[0]
        return linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
    else:
        coefs = np.empty([n_targets, n_features], dtype=X.dtype)
        for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
            A.flat[:: n_features + 1] += current_alpha
            coef[:] = linalg.solve(A, target, assume_a="pos", overwrite_a=False).ravel()
            A.flat[:: n_features + 1] -= current_alpha
        return coefs
