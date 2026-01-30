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

    def _mul(self, other: PolyElement[Er]) -> PolyElement[Er]:
        ring = self.ring
        p = ring.zero
        for exp1, v1 in self.iterterms():
            for exp2, v2 in other.iterterms():
                exp = ring.monomial_mul(exp1, exp2)
                v = v1 * v2
                p[exp] = p.get(exp, ring.domain.zero) + v
        p.strip_zero()
        return p

    def iterterms(self) -> Iterator[tuple[Mon, Er]]:
        """Iterator over terms of a polynomial."""
        return iter(self.items())
