from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.matrices.kind import MatrixKind
from .matpow import MatPow

class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    __slots__: tuple[str, ...] = ()
    _iterable = False
    _op_priority = 11.0
    is_Matrix: bool = True
    is_MatrixExpr: bool = True
    is_Identity: FuzzyBool = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False
    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False
    kind: MatrixKind = MatrixKind()

    def _eval_power(self, exp):
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
        return MatPow(self, exp)
