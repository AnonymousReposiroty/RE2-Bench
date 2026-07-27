from .intfunc import num_digits, igcd, ilcm, mod_inverse, integer_nthroot
from .parameters import global_parameters

class Integer(Rational):
    """Represents integer numbers of any size.

    Examples
    ========

    >>> from sympy import Integer
    >>> Integer(3)
    3

    If a float or a rational is passed to Integer, the fractional part
    will be discarded; the effect is of rounding toward zero.

    >>> Integer(3.8)
    3
    >>> Integer(-3.8)
    -3

    A string is acceptable input if it can be parsed as an integer:

    >>> Integer("9" * 20)
    99999999999999999999

    It is rarely needed to explicitly instantiate an Integer, because
    Python integers are automatically converted to Integer when they
    are used in SymPy expressions.
    """
    q = 1
    is_integer = True
    is_number = True
    is_Integer = True
    __slots__ = ()

    def __rmul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other * self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p * self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)
