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
from sympy.polys.polyutils import (
    expr_from_dict,
    _dict_reorder,
    _parallel_dict_from_expr,
)
from sympy.printing.defaults import DefaultPrinting

class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict[tuple[int, ...], Er], Generic[Er]):
    """Element of multivariate distributed polynomial ring."""

    def __init__(self, ring: PolyRing[Er], init: dict[Mon, Er] | Iterable[tuple[Mon, Er]]):
        super().__init__(init)
        self.ring = ring
    _hash = None
    rem_ground = trunc_ground

    def _change_ring(self, new_ring):
        if self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)
