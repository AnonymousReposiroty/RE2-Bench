from typing import Type

class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __hash__(self):
        """hash - same as mpq/Fraction"""
        try:
            dinv = pow(self.denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            hash_ = _PyHASH_INF
        else:
            hash_ = hash(hash(abs(self.numerator)) * dinv)
        result = hash_ if self.numerator >= 0 else -hash_
        return -2 if result == -1 else result
    _compatible_types: tuple[Type, ...] = ()
