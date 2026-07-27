from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices.kind import MatrixKind
from sympy.utilities.misc import filldedent
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.slice import MatrixSlice

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

    def _entry(self, i, j, **kwargs):
        raise NotImplementedError('Indexing not implemented for %s' % self.__class__.__name__)

    def valid_index(self, i, j):

        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return is_valid(i) and is_valid(j) and (self.rows is None or ((i >= -self.rows) != False and (i < self.rows) != False)) and ((j >= -self.cols) != False) and ((j < self.cols) != False)

    def __getitem__(self, key):
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            i, j = (_sympify(i), _sympify(j))
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError('Invalid indices (%s, %s)' % (i, j))
        elif isinstance(key, (SYMPY_INTS, Integer)):
            rows, cols = self.shape
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('\n                    Single indexing is only supported when the number\n                    of columns is known.'))
            key = _sympify(key)
            i = key // cols
            j = key % cols
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError('Invalid index %s' % key)
        elif isinstance(key, (Symbol, Expr)):
            raise IndexError(filldedent('\n                Only integers may be used when addressing the matrix\n                with a single index.'))
        raise IndexError('Invalid index, wanted %s[i,j]' % self)
