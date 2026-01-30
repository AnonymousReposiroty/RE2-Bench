from sympy.external.gmpy import MPQ
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.utilities import public

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

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented
