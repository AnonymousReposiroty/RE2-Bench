from typing import Type

class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
        if isinstance(other, PythonMPQ):
            return self.numerator == other.numerator and self.denominator == other.denominator
        elif isinstance(other, self._compatible_types):
            return self.__eq__(PythonMPQ(other))
        else:
            return NotImplemented
    _compatible_types: tuple[Type, ...] = ()
