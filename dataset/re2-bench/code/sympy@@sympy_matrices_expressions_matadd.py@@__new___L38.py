from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.exceptions import sympy_deprecation_warning

class MatAdd(MatrixExpr, Add):
    """A Sum of Matrix Expressions

    MatAdd inherits from and operates like SymPy Add

    Examples
    ========

    >>> from sympy import MatAdd, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> C = MatrixSymbol('C', 5, 5)
    >>> MatAdd(A, B, C)
    A + B + C
    """
    is_MatAdd = True
    identity = GenericZeroMatrix()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        if not all((isinstance(arg, MatrixExpr) for arg in args)):
            raise TypeError('Mix of Matrix and Scalar symbols')
        obj = Basic.__new__(cls, *args)
        if check is not None:
            sympy_deprecation_warning('Passing check to MatAdd is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*args)
        if evaluate:
            obj = cls._evaluate(obj)
        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)
