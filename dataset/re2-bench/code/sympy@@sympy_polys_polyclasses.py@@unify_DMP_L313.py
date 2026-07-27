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
from sympy.polys.polyerrors import (
    UnificationFailed,
    PolynomialError)
from typing import Self, TypeAlias

class DMP(CantSympify, Generic[Er]):
    """Dense Multivariate Polynomials over `K`. """
    __slots__ = ()
    lev: int
    dom: Domain[Er]

    @overload
    def unify_DMP(f, g: Self) -> tuple[Self, Self]:
        ...

    @overload
    def unify_DMP(f, g: DMP[Es]) -> tuple[DMP[Et], DMP[Et]]:
        ...

    def unify_DMP(f, g: DMP[Es]) -> tuple[DMP[Et], DMP[Et]]:
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if f.dom == g.dom:
            return (f, g)
        else:
            dom: Domain[Et] = f.dom.unify(g.dom)
            return (f.convert(dom), g.convert(dom))
