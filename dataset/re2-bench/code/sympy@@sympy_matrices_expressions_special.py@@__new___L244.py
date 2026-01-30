from sympy.core.relational import Eq
from sympy.core.sympify import _sympify
from .matexpr import MatrixExpr

class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.

    Also called "matrix of ones" or "all-ones matrix".

    https://en.wikipedia.org/wiki/Matrix_of_ones

    Examples
    ========

    >>> from sympy.matrices.expressions import OneMatrix
    >>> O = OneMatrix(3, 4)
    >>> O.shape
    (3, 4)
    >>> O.as_explicit()
    Matrix([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]])
    """

    def __new__(cls, m, n, evaluate=False):
        m, n = (_sympify(m), _sympify(n))
        cls._check_dim(m)
        cls._check_dim(n)
        if evaluate:
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                return Identity(1)
        obj = super().__new__(cls, m, n)
        return obj
