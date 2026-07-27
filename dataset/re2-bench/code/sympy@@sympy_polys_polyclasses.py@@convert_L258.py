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
import flint

class DMP(CantSympify, Generic[Er]):
    """Dense Multivariate Polynomials over `K`. """
    __slots__ = ()
    lev: int
    dom: Domain[Er]

    def convert(f, dom: Domain[Es]) -> DMP[Es]:
        """Convert ``f`` to a ``DMP`` over the new domain. """
        if f.dom == dom:
            return f
        elif f.lev or flint is None:
            return f._convert(dom)
        elif isinstance(f, DUP_Flint):
            if _supported_flint_domain(dom):
                return f._convert(dom)
            else:
                return f.to_DMP_Python()._convert(dom)
        elif isinstance(f, DMP_Python):
            if _supported_flint_domain(dom):
                return f._convert(dom).to_DUP_Flint()
            else:
                return f._convert(dom)
        else:
            raise RuntimeError('unreachable code')
