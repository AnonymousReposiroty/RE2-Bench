from .matexpr import MatrixExpr
from sympy.core.sympify import _sympify
from sympy.matrices.exceptions import NonSquareMatrixError

class MatPow(MatrixExpr):

    def __new__(cls, base, exp, evaluate=False, **options) -> MatrixExpr:
        base = _sympify(base)
        if not base.is_Matrix:
            raise TypeError('MatPow base should be a matrix')
        if base.is_square is False:
            raise NonSquareMatrixError('Power of non-square matrix %s' % base)
        exp = _sympify(exp)
        obj = super().__new__(cls, base, exp)
        if evaluate:
            obj = obj.doit(deep=False)
        return obj
