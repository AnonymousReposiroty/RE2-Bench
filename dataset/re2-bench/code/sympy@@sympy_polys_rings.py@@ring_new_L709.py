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
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.compatibility import IPolys
from sympy.polys.densebasic import dup, dmp, dmp_to_dict, dup_from_dict, dmp_from_dict
from sympy.polys.domains.domain import Domain, Er, Es, Et
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.polyerrors import (
    CoercionFailed,
    GeneratorsError,
    ExactQuotientFailed,
    MultivariatePolynomialError,
)
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

    def ground_new(self, coeff) -> PolyElement[Er]:
        """Create a constant polynomial with given coefficient."""
        return self.term_new(self.zero_monom, coeff)

    def term_new(self, monom: Mon, coeff: int | Er) -> PolyElement[Er]:
        """Create a polynomial with a single term."""
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    def from_dict(self, element: Mapping[Mon, int | Er | Expr] | PolyElement[Er], orig_domain: Domain[Er] | None=None) -> PolyElement[Er]:
        """Create polynomial from dictionary of monomials to coefficients."""
        if not isinstance(element, dict):
            raise TypeError('Input must be a dictionary mapping monomials to coefficients')
        return self._from_dict_ground(element, orig_domain)

    def from_terms(self, element: Iterable[tuple[Mon, Er]], orig_domain: Domain[Er] | None=None) -> PolyElement[Er]:
        """Create polynomial from sequence of (monomial, coefficient) pairs."""
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element: dmp[Er]) -> PolyElement[Er]:
        """Create polynomial from list(dense) representation."""
        poly_dict = dmp_to_dict(element, self.ngens - 1, self.domain)
        return self.from_dict(poly_dict)

    def from_expr(self, expr) -> PolyElement[Er]:
        """Create polynomial from SymPy expression."""
        mapping = dict(zip(self.symbols, self.gens))
        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError(f'expected an expression convertible to a polynomial in {self}, got {expr}')
        else:
            return self.ring_new(poly)

    def _rebuild_expr(self, expr, mapping) -> PolyElement[Er]:
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)
            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, map(_rebuild, expr.args))
            elif expr.is_Mul:
                return reduce(mul, map(_rebuild, expr.args))
            else:
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base) ** int(exp)
                else:
                    return self.ground_new(domain.convert(expr))
        return _rebuild(sympify(expr))

    def ring_new(self, element) -> PolyElement[Er]:
        """Create a ring element from various input types."""
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError('conversion')
        elif isinstance(element, str):
            raise NotImplementedError('parsing')
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)
    __call__ = ring_new

    def _from_dict_ground(self, element: Mapping[Mon, int | Er | Expr], orig_domain=None) -> PolyElement[Er]:
        poly = self.zero
        domain_new = self.domain_new
        for monom, coeff in element.items():
            if coeff:
                coeff = domain_new(coeff, orig_domain)
                poly[monom] = coeff
        return poly
