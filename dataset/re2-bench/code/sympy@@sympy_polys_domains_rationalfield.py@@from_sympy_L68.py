from sympy.external.gmpy import MPQ
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains import RR

@public
class RationalField(Field[MPQ], CharacteristicZero, SimpleDomain):
    """Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """
    rep = 'QQ'
    alias = 'QQ'
    is_RationalField = is_QQ = True
    is_Numerical = True
    has_assoc_Ring = True
    has_assoc_Field = True
    dtype = MPQ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)

    def __init__(self):
        pass

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed('expected `Rational` object, got %s' % a)
