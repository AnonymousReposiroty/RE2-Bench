from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.matrices.exceptions import NonSquareMatrixError

class Determinant(Expr):
    """Matrix Determinant

    Represents the determinant of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Determinant, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Determinant(A)
    Determinant(A)
    >>> Determinant(eye(3)).doit()
    1
    """
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError('Input to Determinant, %s, not a matrix' % str(mat))
        if mat.is_square is False:
            raise NonSquareMatrixError('Det of a non-square matrix')
        return Basic.__new__(cls, mat)
