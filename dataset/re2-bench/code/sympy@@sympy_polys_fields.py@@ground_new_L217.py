from typing import Generic, Protocol, Any
from sympy.core.expr import Expr
from sympy.polys.domains.domain import Domain, Er, Es
from sympy.polys.domains.field import Field
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
    __call__ = field_new
