from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.matrices.kind import MatrixKind

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

    @property
    def shape(self) -> tuple[Expr | int, Expr | int]:
        raise NotImplementedError

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def valid_index(self, i, j):

        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return is_valid(i) and is_valid(j) and (self.rows is None or ((i >= -self.rows) != False and (i < self.rows) != False)) and ((j >= -self.cols) != False) and ((j < self.cols) != False)
