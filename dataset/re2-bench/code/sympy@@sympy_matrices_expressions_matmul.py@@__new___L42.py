from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True
    identity = GenericIdentity()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        factor, matrices = obj.as_coeff_matrices()
        if check is not None:
            sympy_deprecation_warning('Passing check to MatMul is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*matrices)
        if not matrices:
            return factor
        if evaluate:
            return cls._evaluate(obj)
        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)
