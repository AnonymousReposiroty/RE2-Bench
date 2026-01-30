from sympy.core.basic import Basic
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.sympify import sympify
from sympy.matrices.exceptions import NonSquareMatrixError

class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)
    >>> Trace(eye(3))
    Trace(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> Trace(eye(3)).simplify()
    3
    """
    is_Trace = True
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError('input to Trace, %s, is not a matrix' % str(mat))
        if mat.is_square is False:
            raise NonSquareMatrixError('Trace of a non-square matrix')
        return Basic.__new__(cls, mat)
