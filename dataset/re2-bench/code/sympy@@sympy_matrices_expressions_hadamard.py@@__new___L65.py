from sympy.core import Mul, sympify
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.exceptions import sympy_deprecation_warning

class HadamardProduct(MatrixExpr):
    """
    Elementwise product of matrix expressions

    Examples
    ========

    Hadamard product for matrix symbols:

    >>> from sympy import hadamard_product, HadamardProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(hadamard_product(A, B), HadamardProduct)
    True

    Notes
    =====

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``hadamard_product()`` or ``HadamardProduct.doit``
    """
    is_HadamardProduct = True

    def __new__(cls, *args, evaluate=False, check=None):
        args = list(map(sympify, args))
        if len(args) == 0:
            raise ValueError('HadamardProduct needs at least one argument')
        if not all((isinstance(arg, MatrixExpr) for arg in args)):
            raise TypeError('Mix of Matrix and Scalar symbols')
        if check is not None:
            sympy_deprecation_warning('Passing check to HadamardProduct is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*args)
        obj = super().__new__(cls, *args)
        if evaluate:
            obj = obj.doit(deep=False)
        return obj
