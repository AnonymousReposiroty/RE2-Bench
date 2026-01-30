from typing import (
    Generic,
    overload,
    Callable,
    Iterable,
    Iterator,
    TYPE_CHECKING,
    Mapping,
    cast,
    Sequence,
)
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.domains.domain import Domain, Er, Es, Et
from sympy.polys.domains.domainelement import DomainElement
from sympy.printing.defaults import DefaultPrinting

class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict[tuple[int, ...], Er], Generic[Er]):
    """Element of multivariate distributed polynomial ring."""

    def __init__(self, ring: PolyRing[Er], init: dict[Mon, Er] | Iterable[tuple[Mon, Er]]):
        super().__init__(init)
        self.ring = ring
    _hash = None
    rem_ground = trunc_ground

    def leading_expv(self) -> Mon | None:
        """Leading monomial tuple according to the monomial ordering.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        try:
            return self._leading_expv()
        except KeyError:
            return None

    def _leading_expv(self) -> Mon:
        if not self:
            raise KeyError
        else:
            return self.ring.leading_expv(self)
