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

def fast_logdet(A):
    """Compute logarithm of determinant of a square matrix.

    The (natural) logarithm of the determinant of a square matrix
    is returned if det(A) is non-negative and well defined.
    If the determinant is zero or negative returns -Inf.

    Equivalent to : np.log(np.det(A)) but more robust.

    Parameters
    ----------
    A : array_like of shape (n, n)
        The square matrix.

    Returns
    -------
    logdet : float
        When det(A) is strictly positive, log(det(A)) is returned.
        When det(A) is non-positive or not defined, then -inf is returned.

    See Also
    --------
    numpy.linalg.slogdet : Compute the sign and (natural) logarithm of the determinant
        of an array.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import fast_logdet
    >>> a = np.array([[5, 1], [2, 8]])
    >>> fast_logdet(a)
    np.float64(3.6375861597263857)
    """
    xp, _ = get_namespace(A)
    sign, ld = xp.linalg.slogdet(A)
    if not sign > 0:
        return -xp.inf
    return ld
