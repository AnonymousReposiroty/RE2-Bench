import itertools
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

def _pairwise_callable(X, Y, metric, ensure_all_finite=True, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}."""
    xp, _, device = get_namespace_and_device(X)
    X, Y = check_pairwise_arrays(
        X,
        Y,
        dtype=None,
        ensure_all_finite=ensure_all_finite,
        # No input dimension checking done for custom metrics (left to user)
        ensure_2d=False,
    )
    _, _, dtype_float = _find_floating_dtype_allow_sparse(X, Y, xp=xp)

    def _get_slice(array, index):
        # TODO: below 2 lines can be removed once min scipy >= 1.14. Support for
        # 1D shapes in scipy sparse arrays (COO, DOK and CSR formats) only
        # added in 1.14. We must return 2D array until min scipy 1.14.
        if issparse(array):
            return array[[index], :]
        # When `metric` is a callable, 1D input arrays allowed, in which case
        # scalar should be returned.
        if array.ndim == 1:
            return array[index]
        else:
            return array[index, ...]

    if X is Y:
        # Only calculate metric for upper triangle
        out = xp.zeros((X.shape[0], Y.shape[0]), dtype=dtype_float, device=device)
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            x = _get_slice(X, i)
            y = _get_slice(Y, j)
            out[i, j] = metric(x, y, **kwds)

        # Make symmetric
        # NB: out += out.T will produce incorrect results
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = _get_slice(X, i)
            out[i, i] = metric(x, x, **kwds)

    else:
        # Calculate all cells
        out = xp.empty((X.shape[0], Y.shape[0]), dtype=dtype_float)
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            x = _get_slice(X, i)
            y = _get_slice(Y, j)
            out[i, j] = metric(x, y, **kwds)

    return out
