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

    def __add__(self, other):
        """q1 + q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = (self.numerator, self.denominator)
            bp, bq = (other.numerator, other.denominator)
            g = gcd(aq, bq)
            if g == 1:
                p = ap * bq + aq * bp
                q = bq * aq
            else:
                q1, q2 = (aq // g, bq // g)
                p, q = (ap * q2 + bp * q1, q1 * q2)
                g2 = gcd(p, g)
                p, q = (p // g2, q * (g // g2))
        elif isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
        else:
            return NotImplemented
        return self._new(p, q)
    _compatible_types: tuple[Type, ...] = ()
