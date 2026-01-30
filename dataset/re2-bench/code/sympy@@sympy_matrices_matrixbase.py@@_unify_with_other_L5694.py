from sympy.core.expr import Expr
from typing import Literal, TypeVar, Iterator, Mapping, Any
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.matrixbase import MatrixBase

def _unify_with_other(self: MatrixBase, other: Any
            ) -> tuple[MatrixBase, MatrixBase | Expr, str]:
    """Unify self and other into a single matrix type, or check for scalar."""
    other, T = _coerce_operand(self, other)

    if isinstance(other, MatrixBase):
        typ = classof(self, other)
        if typ != self.__class__:
            self = _convert_matrix(typ, self)
        if typ != other.__class__:
            other = _convert_matrix(typ, other)

    return self, other, T
