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
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __mul__(self, other):
        """q1 * q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = (self.numerator, self.denominator)
            bp, bq = (other.numerator, other.denominator)
            x1 = gcd(ap, bq)
            x2 = gcd(bp, aq)
            p, q = (ap // x1 * (bp // x2), aq // x2 * (bq // x1))
        elif isinstance(other, int):
            x = gcd(other, self.denominator)
            p = self.numerator * (other // x)
            q = self.denominator // x
        else:
            return NotImplemented
        return self._new(p, q)
    _compatible_types: tuple[Type, ...] = ()
