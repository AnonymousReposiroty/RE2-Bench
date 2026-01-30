from sympy.external.gmpy import MPZ, GROUND_TYPES
from sympy.core.numbers import int_valued
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class IntegerRing(Ring[MPZ], CharacteristicZero, SimpleDomain):
    """The domain ``ZZ`` representing the integers `\\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """
    rep = 'ZZ'
    alias = 'ZZ'
    dtype = MPZ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    is_IntegerRing = is_ZZ = True
    is_Numerical = True
    is_PID = True
    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self):
        """Allow instantiation of this domain. """

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed('expected an integer, got %s' % a)
