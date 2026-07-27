from math import gcd
from decimal import Decimal
from fractions import Fraction
from typing import Type

class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __new__(cls, numerator, denominator=None):
        """Construct PythonMPQ with gcd computation and checks"""
        if denominator is not None:
            if isinstance(numerator, int) and isinstance(denominator, int):
                divisor = gcd(numerator, denominator)
                numerator //= divisor
                denominator //= divisor
                return cls._new_check(numerator, denominator)
        else:
            if isinstance(numerator, int):
                return cls._new(numerator, 1)
            elif isinstance(numerator, PythonMPQ):
                return cls._new(numerator.numerator, numerator.denominator)
            if isinstance(numerator, (Decimal, float, str)):
                numerator = Fraction(numerator)
            if isinstance(numerator, Fraction):
                return cls._new(numerator.numerator, numerator.denominator)
        raise TypeError('PythonMPQ() requires numeric or string argument')

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
    _compatible_types: tuple[Type, ...] = ()
