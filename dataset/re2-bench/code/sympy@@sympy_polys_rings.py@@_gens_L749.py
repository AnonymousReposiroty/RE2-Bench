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

    def monomial_basis(self, i) -> tuple[int, ...]:
        """Return the i-th basis element."""
        basis = [0] * self.ngens
        basis[i] = 1
        return tuple(basis)
    __call__ = ring_new

    def _gens(self) -> tuple[PolyElement[Er], ...]:
        one = self.domain.one
        generators = []
        for i in range(self.ngens):
            expv = self.monomial_basis(i)
            poly = self.zero
            poly[expv] = one
            generators.append(poly)
        return tuple(generators)
