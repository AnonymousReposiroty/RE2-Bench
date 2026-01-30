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

    def new(self, init) -> PolyElement[Er]:
        """Create a new polynomial element in the same ring."""
        return self.__class__(self.ring, init)
    rem_ground = trunc_ground

    def clear_denoms(self) -> tuple[Er, PolyElement[Er]]:
        """Clear denominators from polynomial coefficients."""
        domain = self.ring.domain
        if not domain.is_Field or not domain.has_assoc_Ring:
            return (domain.one, self)
        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom
        for coeff in self.values():
            common = lcm(common, denom(coeff))
        poly = self.new([(monom, coeff * common) for monom, coeff in self.items()])
        return (common, poly)
