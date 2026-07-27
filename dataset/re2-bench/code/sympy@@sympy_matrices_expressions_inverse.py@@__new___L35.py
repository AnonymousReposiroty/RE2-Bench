from sympy.core.sympify import _sympify
from sympy.core import S, Basic
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions.matpow import MatPow

class Inverse(MatPow):
    """
    The multiplicative inverse of a matrix expression

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the inverse, use the ``.inverse()``
    method of matrices.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Inverse
    >>> A = MatrixSymbol('A', 3, 3)
    >>> B = MatrixSymbol('B', 3, 3)
    >>> Inverse(A)
    A**(-1)
    >>> A.inverse() == Inverse(A)
    True
    >>> (A*B).inverse()
    B**(-1)*A**(-1)
    >>> Inverse(A*B)
    (A*B)**(-1)

    """
    is_Inverse = True
    exp = S.NegativeOne

    def __new__(cls, mat, exp=S.NegativeOne):
        mat = _sympify(mat)
        exp = _sympify(exp)
        if not mat.is_Matrix:
            raise TypeError('mat should be a matrix')
        if mat.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix %s' % mat)
        return Basic.__new__(cls, mat, exp)
