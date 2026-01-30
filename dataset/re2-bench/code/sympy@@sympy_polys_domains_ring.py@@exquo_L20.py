from sympy.polys.domains.domain import Domain, Er
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible
from sympy.utilities import public

@public
class Ring(Domain[Er]):
    """Represents a ring domain. """
    is_Ring = True

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__floordiv__``.  """
        if a % b:
            raise ExactQuotientFailed(a, b, self)
        else:
            return a // b
