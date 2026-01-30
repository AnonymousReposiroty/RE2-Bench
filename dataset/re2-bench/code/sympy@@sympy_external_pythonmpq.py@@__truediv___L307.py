from math import gcd
from typing import Type

class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
        if not denominator:
            raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
        elif denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return cls._new(numerator, denominator)

    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __truediv__(self, other):
        """q1 / q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = (self.numerator, self.denominator)
            bp, bq = (other.numerator, other.denominator)
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = (ap // x1 * (bq // x2), aq // x2 * (bp // x1))
        elif isinstance(other, int):
            x = gcd(other, self.numerator)
            p = self.numerator // x
            q = self.denominator * (other // x)
        else:
            return NotImplemented
        return self._new_check(p, q)
    _compatible_types: tuple[Type, ...] = ()
