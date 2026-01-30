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
from sympy.polys.compatibility import IPolys
from sympy.polys.domains.domain import Domain, Er, Es, Et
from sympy.polys.orderings import lex, MonomialOrder
from sympy.printing.defaults import DefaultPrinting

class PolyRing(DefaultPrinting, IPolys[Er], Generic[Er]):
    """Multivariate distributed polynomial ring."""
    symbols: tuple[Expr, ...]
    gens: tuple[PolyElement[Er], ...]
    ngens: int
    _gens_set: set[PolyElement]
    domain: Domain[Er]
    order: MonomialOrder
    _hash: int
    _hash_tuple: tuple
    _one: list[tuple[Mon, Er]]
    dtype: Callable[[Iterable[tuple[Mon, Er]] | dict[Mon, Er]], PolyElement[Er]]
    monomial_mul: Callable[[Mon, Mon], Mon]
    monomial_pow: Callable[[Mon, int], Mon]
    monomial_mulpow: Callable[[Mon, Mon, int], Mon]
    monomial_ldiv: Callable[[Mon, Mon], Mon]
    monomial_div: Callable[[Mon, Mon], Mon]
    monomial_lcm: Callable[[Mon, Mon], Mon]
    monomial_gcd: Callable[[Mon, Mon], Mon]
    leading_expv: Callable[[PolyElement[Er]], Mon]
    zero_monom: Mon

    @property
    def zero(self) -> PolyElement[Er]:
        """The zero polynomial."""
        return self.dtype([])

    def domain_new(self, element, orig_domain=None) -> Er:
        """Create a new element of the ground domain."""
        return self.domain.convert(element, orig_domain)

    def term_new(self, monom: Mon, coeff: int | Er) -> PolyElement[Er]:
        """Create a polynomial with a single term."""
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly
    __call__ = ring_new
