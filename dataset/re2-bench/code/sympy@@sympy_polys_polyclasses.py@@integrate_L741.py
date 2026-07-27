from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    overload,
    Callable,
    TypeVar,
)
from sympy.core.sympify import CantSympify
from sympy.polys.domains import Domain, ZZ, QQ
from sympy.polys.domains.domain import Er, Es, Et, Eg
from typing import Self, TypeAlias

class DMP(CantSympify, Generic[Er]):
    """Dense Multivariate Polynomials over `K`. """
    __slots__ = ()
    lev: int
    dom: Domain[Er]

    def integrate(f, m: int=1, j: int=0) -> Self:
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError('``int`` expected, got %s' % type(m))
        if not isinstance(j, int):
            raise TypeError('``int`` expected, got %s' % type(j))
        return f._integrate(m, j)
