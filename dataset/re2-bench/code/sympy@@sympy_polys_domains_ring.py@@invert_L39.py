from sympy.polys.domains.domain import Domain, Er
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible
from sympy.utilities import public

@public
class Ring(Domain[Er]):
    """Represents a ring domain. """
    is_Ring = True

    def invert(self, a, b):
        """Returns inversion of ``a mod b``. """
        s, t, h = self.gcdex(a, b)
        if self.is_one(h):
            return s % b
        else:
            raise NotInvertible('zero divisor')
