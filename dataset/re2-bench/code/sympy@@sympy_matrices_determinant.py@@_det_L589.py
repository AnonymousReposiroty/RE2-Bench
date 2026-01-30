from types import FunctionType
from sympy.core.mul import Mul
from .utilities import (
    _get_intermediate_simp, _get_intermediate_simp_bool,
    _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify)

def _det(M, method="bareiss", iszerofunc=None):
    """Computes the determinant of a matrix if ``M`` is a concrete matrix object
    otherwise return an expressions ``Determinant(M)`` if ``M`` is a
    ``MatrixSymbol`` or other expression.

    Parameters
    ==========

    method : string, optional
        Specifies the algorithm used for computing the matrix determinant.

        If the matrix is at most 3x3, a hard-coded formula is used and the
        specified method is ignored. Otherwise, it defaults to
        ``'bareiss'``.

        Also, if the matrix is an upper or a lower triangular matrix, determinant
        is computed by simple multiplication of diagonal elements, and the
        specified method is ignored.

        If it is set to ``'domain-ge'``, then Gaussian elimination method will
        be used via using DomainMatrix.

        If it is set to ``'bareiss'``, Bareiss' fraction-free algorithm will
        be used.

        If it is set to ``'berkowitz'``, Berkowitz' algorithm will be used.

        If it is set to ``'bird'``, Bird's algorithm will be used [1]_.

        If it is set to ``'laplace'``, Laplace's algorithm will be used.

        Otherwise, if it is set to ``'lu'``, LU decomposition will be used.

        .. note::
            For backward compatibility, legacy keys like "bareis" and
            "det_lu" can still be used to indicate the corresponding
            methods.
            And the keys are also case-insensitive for now. However, it is
            suggested to use the precise keys for specifying the method.

    iszerofunc : FunctionType or None, optional
        If it is set to ``None``, it will be defaulted to ``_iszero`` if the
        method is set to ``'bareiss'``, and ``_is_zero_after_expand_mul`` if
        the method is set to ``'lu'``.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        tested as non-zero, and also ``None`` if it is undecidable.

    Returns
    =======

    det : Basic
        Result of determinant.

    Raises
    ======

    ValueError
        If unrecognized keys are given for ``method`` or ``iszerofunc``.

    NonSquareMatrixError
        If attempted to calculate determinant from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix, eye, det
    >>> I3 = eye(3)
    >>> det(I3)
    1
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> det(M)
    -2
    >>> det(M) == M.det()
    True
    >>> M.det(method="domain-ge")
    -2

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """

    # sanitize `method`
    method = method.lower()

    if method == "bareis":
        method = "bareiss"
    elif method == "det_lu":
        method = "lu"

    if method not in ("bareiss", "berkowitz", "lu", "domain-ge", "bird",
                      "laplace"):
        raise ValueError("Determinant method '%s' unrecognized" % method)

    if iszerofunc is None:
        if method == "bareiss":
            iszerofunc = _is_zero_after_expand_mul
        elif method == "lu":
            iszerofunc = _iszero

    elif not isinstance(iszerofunc, FunctionType):
        raise ValueError("Zero testing method '%s' unrecognized" % iszerofunc)

    n = M.rows

    if n == M.cols: # square check is done in individual method functions
        if n == 0:
            return M.one
        elif n == 1:
            return M[0, 0]
        elif n == 2:
            m = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            return _get_intermediate_simp(_dotprodsimp)(m)
        elif n == 3:
            m =  (M[0, 0] * M[1, 1] * M[2, 2]
                + M[0, 1] * M[1, 2] * M[2, 0]
                + M[0, 2] * M[1, 0] * M[2, 1]
                - M[0, 2] * M[1, 1] * M[2, 0]
                - M[0, 0] * M[1, 2] * M[2, 1]
                - M[0, 1] * M[1, 0] * M[2, 2])
            return _get_intermediate_simp(_dotprodsimp)(m)

    dets = []
    for b in M.strongly_connected_components():
        if method == "domain-ge": # uses DomainMatrix to evaluate determinant
            det = _det_DOM(M[b, b])
        elif method == "bareiss":
            det = M[b, b]._eval_det_bareiss(iszerofunc=iszerofunc)
        elif method == "berkowitz":
            det = M[b, b]._eval_det_berkowitz()
        elif method == "lu":
            det = M[b, b]._eval_det_lu(iszerofunc=iszerofunc)
        elif method == "bird":
            det = M[b, b]._eval_det_bird()
        elif method == "laplace":
            det = M[b, b]._eval_det_laplace()
        else:
            assert False
        dets.append(det)
    return Mul(*dets)
