from typing import Generic, Protocol, Any
from sympy.core.sympify import CantSympify, sympify
from sympy.polys.domains.domain import Domain, Er, Es
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.rings import PolyRing, PolyElement
from sympy.printing.defaults import DefaultPrinting

class FracElement(DomainElement, DefaultPrinting, CantSympify, Generic[Er]):
    """Element of multivariate distributed rational function field. """

    def __init__(self, field: FracField[Er], numer: PolyElement[Er], denom: PolyElement[Er] | None=None) -> None:
        if denom is None:
            denom = field.ring.one
        elif not denom:
            raise ZeroDivisionError('zero denominator')
        self.field = field
        self.numer = numer
        self.denom = denom
    _hash = None

    def __truediv__(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        field = f.field
        if not g:
            raise ZeroDivisionError
        elif field.is_element(g):
            return f.new(f.numer * g.denom, f.denom * g.numer)
        elif field.ring.is_element(g):
            return f.new(f.numer, f.denom * g)
        elif isinstance(g, FracElement):
            if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                pass
            elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                return g.__rtruediv__(f)
            else:
                return NotImplemented
        elif isinstance(g, PolyElement):
            if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                pass
            else:
                return g.__rtruediv__(f)
        op, g_numer, g_denom = f._extract_ground(g)
        if op == 1:
            return f.new(f.numer, f.denom * g_numer)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer * g_denom, f.denom * g_numer)
