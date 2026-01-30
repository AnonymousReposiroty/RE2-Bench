from typing import Generic, TYPE_CHECKING
from sympy.polys.domains.domain import Er
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.field import Field
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public
from sympy.polys.domains.domain import Domain
from sympy.polys.fields import FracField, FracElement
from sympy.polys.fields import FracField

@public
class FractionField(Field, CompositeDomain, Generic[Er]):
    """A class for representing multivariate rational function fields. """
    is_FractionField = is_Frac = True
    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self, domain_or_field: FracField[Er] | Domain[Er], symbols=None, order=None):
        from sympy.polys.fields import FracField
        if isinstance(domain_or_field, FracField) and symbols is None and (order is None):
            field = domain_or_field
        else:
            field = FracField(symbols, domain_or_field, order)
        self.field: FracField[Er] = field
        self.dtype = field.dtype
        self.gens = field.gens
        self.ngens = field.ngens
        self.symbols = field.symbols
        self.domain = field.domain
        self.dom = self.domain

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.is_ground:
            return K1.convert_from(a.coeff(1), K0.domain)
        try:
            return K1.new(a.set_ring(K1.field.ring))
        except (CoercionFailed, GeneratorsError):
            try:
                return K1.new(a)
            except (CoercionFailed, GeneratorsError):
                return None
