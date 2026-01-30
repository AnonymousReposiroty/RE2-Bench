from sympy.core.numbers import I
from sympy.core.expr import Expr
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.ringextension import RingExtension

class GaussianDomain(RingExtension[Telem, Tdom]):
    """Base class for Gaussian domains."""
    dom: Domain
    units: tuple[Telem, Telem, Telem, Telem]
    is_Numerical = True
    is_Exact = True
    has_assoc_Ring = True
    has_assoc_Field = True

    def from_sympy(self, a: Expr) -> Telem:
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)
        if not b:
            return self.new(x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)
        if b is I:
            return self.new(x, y)
        else:
            raise CoercionFailed('{} is not Gaussian'.format(a))
