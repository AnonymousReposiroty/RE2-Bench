from .singleton import S, Singleton
from .expr import Expr, AtomicExpr
from .kind import NumberKind

class Number(AtomicExpr):
    """Represents atomic numbers in SymPy.

    Explanation
    ===========

    Floating point numbers are represented by the Float class.
    Rational numbers (of any size) are represented by the Rational class.
    Integer numbers (of any size) are represented by the Integer class.
    Float and Rational are subclasses of Number; Integer is a subclass
    of Rational.

    For example, ``2/3`` is represented as ``Rational(2, 3)`` which is
    a different object from the floating point number obtained with
    Python division ``2/3``. Even for numbers that are exactly
    represented in binary, there is a difference between how two forms,
    such as ``Rational(1, 2)`` and ``Float(0.5)``, are used in SymPy.
    The rational form is to be preferred in symbolic computations.

    Other kinds of numbers, such as algebraic numbers ``sqrt(2)`` or
    complex numbers ``3 + 4*I``, are not instances of Number class as
    they are not atomic.

    See Also
    ========

    Float, Integer, Rational
    """
    is_commutative = True
    is_number = True
    is_Number = True
    __slots__ = ()
    _prec = -1
    kind = NumberKind

    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if self.is_Rational or not rational:
            return (self, ())
        elif self.is_negative:
            return (S.NegativeOne, (-self,))
        return (S.One, (self,))
