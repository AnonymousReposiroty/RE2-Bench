import numpy as np
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

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    xp, _ = get_namespace(*[a for a in [u, v] if a is not None])

    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = xp.argmax(xp.abs(u.T), axis=1)
        shift = xp.arange(u.T.shape[0], device=device(u))
        indices = max_abs_u_cols + shift * u.T.shape[1]
        signs = xp.sign(xp.take(xp.reshape(u.T, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        if v is not None:
            v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_v_rows = xp.argmax(xp.abs(v), axis=1)
        shift = xp.arange(v.shape[0], device=device(v))
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = xp.sign(xp.take(xp.reshape(v, (-1,)), indices, axis=0))
        if u is not None:
            u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    return u, v
