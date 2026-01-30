from collections.abc import Iterable, Mapping
from .basic import Basic, Atom
from .containers import Tuple
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase
from .relational import Eq
from sympy.functions.elementary.piecewise import Piecewise

class AtomicExpr(Atom, Expr):
    """
    A parent class for object which are both atoms and Exprs.

    For example: Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """
    is_number = False
    is_Atom = True
    __slots__ = ()

    def _eval_derivative_n_times(self, s, n):
        from .containers import Tuple
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(s, (MatrixBase, Tuple, Iterable, MatrixExpr)):
            return super()._eval_derivative_n_times(s, n)
        from .relational import Eq
        from sympy.functions.elementary.piecewise import Piecewise
        if self == s:
            return Piecewise((self, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
        else:
            return Piecewise((self, Eq(n, 0)), (0, True))
