from typing import Generic, Protocol, Any
from functools import reduce
from operator import add, mul, lt, le, gt, ge
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domain import Domain, Er, Es
from sympy.polys.domains.field import Field
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.rings import PolyRing, PolyElement
from sympy.printing.defaults import DefaultPrinting

class FracField(DefaultPrinting, Generic[Er]):
    """Multivariate distributed rational function field. """
    ring: PolyRing[Er]
    gens: tuple[FracElement[Er], ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain[Er]
    order: MonomialOrder
    zero: FracElement[Er]
    one: FracElement[Er]
    dtype: FracElementConstructor[Er]
    _hash: int
    _hash_tuple: Any

    def raw_new(self, numer: PolyElement[Er], denom: PolyElement[Er] | None=None) -> FracElement[Er]:
        return self.dtype(numer, denom)

    def new(self, numer: PolyElement[Er], denom: PolyElement[Er] | None=None) -> FracElement[Er]:
        if denom is None:
            denom = self.ring.one
        numer, denom = numer.cancel(denom)
        return self.raw_new(numer, denom)

    def ground_new(self, element) -> FracElement[Er]:
        try:
            return self.new(self.ring.ground_new(element))
        except CoercionFailed:
            domain = self.domain
            if not domain.is_Field and domain.has_assoc_Field:
                ring = self.ring
                ground_field: Field = domain.get_field()
                element = ground_field.convert(element)
                numer = ring.ground_new(ground_field.numer(element))
                denom = ring.ground_new(ground_field.denom(element))
                return self.raw_new(numer, denom)
            else:
                raise

    def field_new(self, element) -> FracElement[Er]:
        if isinstance(element, FracElement):
            if self == element.field:
                return element
            if isinstance(self.domain, FractionField) and self.domain.field == element.field:
                return self.ground_new(element)
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring.to_field() == element.field:
                return self.ground_new(element)
            else:
                raise NotImplementedError('conversion')
        elif isinstance(element, PolyElement):
            denom, numer = element.clear_denoms()
            if isinstance(self.domain, PolynomialRing) and numer.ring == self.domain.ring:
                numer = self.ring.ground_new(numer)
            elif isinstance(self.domain, FractionField) and numer.ring == self.domain.field.to_ring():
                numer = self.ring.ground_new(numer)
            else:
                numer = numer.set_ring(self.ring)
            denom = self.ring.ground_new(denom)
            return self.raw_new(numer, denom)
        elif isinstance(element, tuple) and len(element) == 2:
            numer, denom = list(map(self.ring.ring_new, element))
            return self.new(numer, denom)
        elif isinstance(element, str):
            raise NotImplementedError('parsing')
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)
    __call__ = field_new

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain
        powers = tuple(((gen, gen.as_base_exp()) for gen in mapping.keys() if gen.is_Pow or isinstance(gen, ExpBase)))

        def _rebuild(expr):
            generator = mapping.get(expr)
            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            elif expr.is_Pow or isinstance(expr, (ExpBase, Exp1)):
                b, e = expr.as_base_exp()
                for gen, (bg, eg) in powers:
                    if bg == b and Mod(e, eg) == 0:
                        return mapping.get(gen) ** int(e / eg)
                if e.is_Integer and e is not S.One:
                    return _rebuild(b) ** int(e)
            elif mapping.get(1 / expr) is not None:
                return 1 / mapping.get(1 / expr)
            try:
                return domain.convert(expr)
            except CoercionFailed:
                if not domain.is_Field and domain.has_assoc_Field:
                    return domain.get_field().convert(expr)
                else:
                    raise
        return _rebuild(expr)

    def from_expr(self, expr: Expr) -> FracElement[Er]:
        mapping = dict(list(zip(self.symbols, self.gens)))
        try:
            frac = self._rebuild_expr(sympify(expr), mapping)
        except CoercionFailed:
            raise ValueError('expected an expression convertible to a rational function in %s, got %s' % (self, expr))
        else:
            return self.field_new(frac)
