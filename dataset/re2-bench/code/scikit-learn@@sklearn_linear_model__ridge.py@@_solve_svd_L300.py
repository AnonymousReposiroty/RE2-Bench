from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)

def _solve_svd(X, y, alpha, xp=None):
    xp, _ = get_namespace(X, xp=xp)
    U, s, Vt = xp.linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, None]
    UTy = U.T @ y
    d = xp.zeros((s.shape[0], alpha.shape[0]), dtype=X.dtype, device=device(X))
    d[idx] = s_nnz / (s_nnz**2 + alpha)
    d_UT_y = d * UTy
    return (Vt.T @ d_UT_y).T
