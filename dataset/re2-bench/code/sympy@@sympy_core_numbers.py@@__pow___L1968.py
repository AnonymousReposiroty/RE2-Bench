from sympy.utilities.misc import debug, as_int

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

    def __pow__(self, other, mod=None):
        if mod is not None:
            try:
                other_int = as_int(other)
                mod_int = as_int(mod)
            except ValueError:
                pass
            else:
                return Integer(pow(self.p, other_int, mod_int))
        return super().__pow__(other, mod)
