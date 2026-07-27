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
from sympy.core.expr import Expr
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

    def as_expr(self, *symbols: Expr) -> Expr:
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError('Wrong number of symbols, expected %s got %s' % (self.ring.ngens, len(symbols)))
        return expr_from_dict(self.as_expr_dict(), *symbols)
    rem_ground = trunc_ground

    def as_expr_dict(self) -> dict[tuple[int, ...], Expr]:
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def iterterms(self) -> Iterator[tuple[Mon, Er]]:
        """Iterator over terms of a polynomial."""
        return iter(self.items())
