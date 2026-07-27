from typing import Any, Generic, TypeVar, Protocol, Callable, Iterable, TYPE_CHECKING
from sympy.core.sorting import ordered
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains import ZZ, QQ, RealField, ComplexField
from sympy.polys.domains import EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.complexfield import ComplexField

@public
class Domain(Generic[Er]):
    """Superclass for all domains in the polys domains system.

    See :ref:`polys-domainsintro` for an introductory explanation of the
    domains system.

    The :py:class:`~.Domain` class is an abstract base class for all of the
    concrete domain types. There are many different :py:class:`~.Domain`
    subclasses each of which has an associated ``dtype`` which is a class
    representing the elements of the domain. The coefficients of a
    :py:class:`~.Poly` are elements of a domain which must be a subclass of
    :py:class:`~.Domain`.

    Examples
    ========

    The most common example domains are the integers :ref:`ZZ` and the
    rationals :ref:`QQ`.

    >>> from sympy import Poly, symbols, Domain
    >>> x, y = symbols('x, y')
    >>> p = Poly(x**2 + y)
    >>> p
    Poly(x**2 + y, x, y, domain='ZZ')
    >>> p.domain
    ZZ
    >>> isinstance(p.domain, Domain)
    True
    >>> Poly(x**2 + y/2)
    Poly(x**2 + 1/2*y, x, y, domain='QQ')

    The domains can be used directly in which case the domain object e.g.
    (:ref:`ZZ` or :ref:`QQ`) can be used as a constructor for elements of
    ``dtype``.

    >>> from sympy import ZZ, QQ
    >>> ZZ(2)
    2
    >>> ZZ.dtype  # doctest: +SKIP
    <class 'int'>
    >>> type(ZZ(2))  # doctest: +SKIP
    <class 'int'>
    >>> QQ(1, 2)
    1/2
    >>> type(QQ(1, 2))  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    The corresponding domain elements can be used with the arithmetic
    operations ``+,-,*,**`` and depending on the domain some combination of
    ``/,//,%`` might be usable. For example in :ref:`ZZ` both ``//`` (floor
    division) and ``%`` (modulo division) can be used but ``/`` (true
    division) cannot. Since :ref:`QQ` is a :py:class:`~.Field` its elements
    can be used with ``/`` but ``//`` and ``%`` should not be used. Some
    domains have a :py:meth:`~.Domain.gcd` method.

    >>> ZZ(2) + ZZ(3)
    5
    >>> ZZ(5) // ZZ(2)
    2
    >>> ZZ(5) % ZZ(2)
    1
    >>> QQ(1, 2) / QQ(2, 3)
    3/4
    >>> ZZ.gcd(ZZ(4), ZZ(2))
    2
    >>> QQ.gcd(QQ(2,7), QQ(5,3))
    1/21
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    There are also many other domains including:

        1. :ref:`GF(p)` for finite fields of prime order.
        2. :ref:`RR` for real (floating point) numbers.
        3. :ref:`CC` for complex (floating point) numbers.
        4. :ref:`QQ(a)` for algebraic number fields.
        5. :ref:`K[x]` for polynomial rings.
        6. :ref:`K(x)` for rational function fields.
        7. :ref:`EX` for arbitrary expressions.

    Each domain is represented by a domain object and also an implementation
    class (``dtype``) for the elements of the domain. For example the
    :ref:`K[x]` domains are represented by a domain object which is an
    instance of :py:class:`~.PolynomialRing` and the elements are always
    instances of :py:class:`~.PolyElement`. The implementation class
    represents particular types of mathematical expressions in a way that is
    more efficient than a normal SymPy expression which is of type
    :py:class:`~.Expr`. The domain methods :py:meth:`~.Domain.from_sympy` and
    :py:meth:`~.Domain.to_sympy` are used to convert from :py:class:`~.Expr`
    to a domain element and vice versa.

    >>> from sympy import Symbol, ZZ, Expr
    >>> x = Symbol('x')
    >>> K = ZZ[x]           # polynomial ring domain
    >>> K
    ZZ[x]
    >>> type(K)             # class of the domain
    <class 'sympy.polys.domains.polynomialring.PolynomialRing'>
    >>> K.dtype             # doctest: +SKIP
    <class 'sympy.polys.rings.PolyElement'>
    >>> p_expr = x**2 + 1   # Expr
    >>> p_expr
    x**2 + 1
    >>> type(p_expr)
    <class 'sympy.core.add.Add'>
    >>> isinstance(p_expr, Expr)
    True
    >>> p_domain = K.from_sympy(p_expr)
    >>> p_domain            # domain element
    x**2 + 1
    >>> type(p_domain)
    <class 'sympy.polys.rings.PolyElement'>
    >>> K.to_sympy(p_domain) == p_expr
    True

    The :py:meth:`~.Domain.convert_from` method is used to convert domain
    elements from one domain to another.

    >>> from sympy import ZZ, QQ
    >>> ez = ZZ(2)
    >>> eq = QQ.convert_from(ez, ZZ)
    >>> type(ez)  # doctest: +SKIP
    <class 'int'>
    >>> type(eq)  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    Elements from different domains should not be mixed in arithmetic or other
    operations: they should be converted to a common domain first.  The domain
    method :py:meth:`~.Domain.unify` is used to find a domain that can
    represent all the elements of two given domains.

    >>> from sympy import ZZ, QQ, symbols
    >>> x, y = symbols('x, y')
    >>> ZZ.unify(QQ)
    QQ
    >>> ZZ[x].unify(QQ)
    QQ[x]
    >>> ZZ[x].unify(QQ[y])
    QQ[x,y]

    If a domain is a :py:class:`~.Ring` then is might have an associated
    :py:class:`~.Field` and vice versa. The :py:meth:`~.Domain.get_field` and
    :py:meth:`~.Domain.get_ring` methods will find or create the associated
    domain.

    >>> from sympy import ZZ, QQ, Symbol
    >>> x = Symbol('x')
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ
    >>> K = QQ[x]
    >>> K
    QQ[x]
    >>> K.get_field()
    QQ(x)

    See also
    ========

    DomainElement: abstract base class for domain elements
    construct_domain: construct a minimal domain for some expressions

    """
    dtype: type[Er] | Callable[..., Er]
    'The type (class) of the elements of this :py:class:`~.Domain`:\n\n    >>> from sympy import ZZ, QQ, Symbol\n    >>> ZZ.dtype\n    <class \'int\'>\n    >>> z = ZZ(2)\n    >>> z\n    2\n    >>> type(z)\n    <class \'int\'>\n    >>> type(z) == ZZ.dtype\n    True\n\n    Every domain has an associated **dtype** ("datatype") which is the\n    class of the associated domain elements.\n\n    See also\n    ========\n\n    of_type\n    '
    zero: Er
    'The zero element of the :py:class:`~.Domain`:\n\n    >>> from sympy import QQ\n    >>> QQ.zero\n    0\n    >>> QQ.of_type(QQ.zero)\n    True\n\n    See also\n    ========\n\n    of_type\n    one\n    '
    one: Er
    'The one element of the :py:class:`~.Domain`:\n\n    >>> from sympy import QQ\n    >>> QQ.one\n    1\n    >>> QQ.of_type(QQ.one)\n    True\n\n    See also\n    ========\n\n    of_type\n    zero\n    '
    is_Ring: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.Ring`.\n\n    >>> from sympy import ZZ\n    >>> ZZ.is_Ring\n    True\n\n    Basically every :py:class:`~.Domain` represents a ring so this flag is\n    not that useful.\n\n    See also\n    ========\n\n    is_PID\n    is_Field\n    get_ring\n    has_assoc_Ring\n    '
    is_Field: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.Field`.\n\n    >>> from sympy import ZZ, QQ\n    >>> ZZ.is_Field\n    False\n    >>> QQ.is_Field\n    True\n\n    See also\n    ========\n\n    is_PID\n    is_Ring\n    get_field\n    has_assoc_Field\n    '
    has_assoc_Ring: bool = False
    'Boolean flag indicating if the domain has an associated\n    :py:class:`~.Ring`.\n\n    >>> from sympy import QQ\n    >>> QQ.has_assoc_Ring\n    True\n    >>> QQ.get_ring()\n    ZZ\n\n    See also\n    ========\n\n    is_Field\n    get_ring\n    '
    has_assoc_Field: bool = False
    'Boolean flag indicating if the domain has an associated\n    :py:class:`~.Field`.\n\n    >>> from sympy import ZZ\n    >>> ZZ.has_assoc_Field\n    True\n    >>> ZZ.get_field()\n    QQ\n\n    See also\n    ========\n\n    is_Field\n    get_field\n    '
    is_FiniteField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.FiniteField`.'
    is_FF: bool = False
    'Alias for :py:attr:`~.Domain.is_FiniteField`.'
    is_IntegerRing: bool = False
    'Boolean flag indicating if the domain is an :py:class:`~.IntegerRing`.'
    is_ZZ: bool = False
    'Alias for :py:attr:`~.Domain.is_IntegerRing`.'
    is_RationalField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.RationalField`.'
    is_QQ: bool = False
    'Alias for :py:attr:`~.Domain.is_RationalField`.'
    is_GaussianRing: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.GaussianIntegerRing`.'
    is_ZZ_I: bool = False
    'Alias for :py:attr:`~.Domain.is_GaussianRing`.'
    is_GaussianField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.GaussianRationalField`.'
    is_QQ_I: bool = False
    'Alias for :py:attr:`~.Domain.is_GaussianField`.'
    is_RealField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.RealField`.'
    is_RR: bool = False
    'Alias for :py:attr:`~.Domain.is_RealField`.'
    is_ComplexField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.ComplexField`.'
    is_CC: bool = False
    'Alias for :py:attr:`~.Domain.is_ComplexField`.'
    is_AlgebraicField: bool = False
    'Boolean flag indicating if the domain is an :py:class:`~.AlgebraicField`.'
    is_Algebraic: bool = False
    'Alias for :py:attr:`~.Domain.is_AlgebraicField`.'
    is_PolynomialRing: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.PolynomialRing`.'
    is_Poly: bool = False
    'Alias for :py:attr:`~.Domain.is_PolynomialRing`.'
    is_FractionField: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.FractionField`.'
    is_Frac: bool = False
    'Alias for :py:attr:`~.Domain.is_FractionField`.'
    is_SymbolicDomain: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.ExpressionDomain`.'
    is_EX: bool = False
    'Alias for :py:attr:`~.Domain.is_SymbolicDomain`.'
    is_SymbolicRawDomain: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.ExpressionRawDomain`.'
    is_EXRAW: bool = False
    'Alias for :py:attr:`~.Domain.is_SymbolicRawDomain`.'
    is_FiniteExtension: bool = False
    'Boolean flag indicating if the domain is a :py:class:`~.MonogenicFiniteExtension`. '
    is_Exact: bool = True
    'Boolean flag indicating if the domain is an exact domain.'
    is_Numerical: bool = False
    'Boolean flag indicating if the domain is a numerical domain.'
    is_Simple: bool = False
    'Boolean flag indicating if the domain is a simple domain.'
    is_Composite: bool = False
    'Boolean flag indicating if the domain is a composite domain.'
    is_RingExtension: bool = False
    'Boolean flag indicating if the domain is a ring extension domain.'
    is_PID: bool = False
    'Boolean flag indicating if the domain is a `principal ideal domain`_.\n\n    >>> from sympy import ZZ\n    >>> ZZ.is_PID\n    True\n\n    .. _principal ideal domain: https://en.wikipedia.org/wiki/Principal_ideal_domain\n\n    See also\n    ========\n\n    is_Field\n    get_field\n    '
    has_CharacteristicZero: bool = False
    'Boolean flag indicating if the domain has characteristic zero.'
    rep: str
    alias: str | None = None

    def __init__(self):
        raise NotImplementedError

    def unify(K0, K1, symbols=None):
        """
        Construct a minimal domain that contains elements of ``K0`` and ``K1``.

        Known domains (from smallest to largest):

        - ``GF(p)``
        - ``ZZ``
        - ``QQ``
        - ``RR(prec, tol)``
        - ``CC(prec, tol)``
        - ``ALG(a, b, c)``
        - ``K[x, y, z]``
        - ``K(x, y, z)``
        - ``EX``

        """
        if symbols is not None:
            return K0.unify_with_symbols(K1, symbols)
        if K0 == K1:
            return K0
        if not (K0.has_CharacteristicZero and K1.has_CharacteristicZero):
            if K0.characteristic() != K1.characteristic():
                raise UnificationFailed('Cannot unify %s with %s' % (K0, K1))
            return K0.unify_composite(K1)
        if K0.is_EXRAW:
            return K0
        if K1.is_EXRAW:
            return K1
        if K0.is_EX:
            return K0
        if K1.is_EX:
            return K1
        if K0.is_FiniteExtension or K1.is_FiniteExtension:
            if K1.is_FiniteExtension:
                K0, K1 = (K1, K0)
            if K1.is_FiniteExtension:
                if list(ordered([K0.modulus, K1.modulus]))[1] == K0.modulus:
                    K0, K1 = (K1, K0)
                return K1.set_domain(K0)
            else:
                K1 = K1.drop(K0.symbol)
                K1 = K0.domain.unify(K1)
                return K0.set_domain(K1)
        if K0.is_Composite or K1.is_Composite:
            return K0.unify_composite(K1)
        if K1.is_ComplexField:
            K0, K1 = (K1, K0)
        if K0.is_ComplexField:
            if K1.is_ComplexField or K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    from sympy.polys.domains.complexfield import ComplexField
                    return ComplexField(prec=K1.precision)
            else:
                return K0
        if K1.is_RealField:
            K0, K1 = (K1, K0)
        if K0.is_RealField:
            if K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    return K1
            elif K1.is_GaussianRing or K1.is_GaussianField:
                from sympy.polys.domains.complexfield import ComplexField
                return ComplexField(prec=K0.precision)
            else:
                return K0
        if K1.is_AlgebraicField:
            K0, K1 = (K1, K0)
        if K0.is_AlgebraicField:
            if K1.is_GaussianRing:
                K1 = K1.get_field()
            if K1.is_GaussianField:
                K1 = K1.as_AlgebraicField()
            if K1.is_AlgebraicField:
                return K0.__class__(K0.dom.unify(K1.dom), *_unify_gens(K0.orig_ext, K1.orig_ext))
            else:
                return K0
        if K0.is_GaussianField:
            return K0
        if K1.is_GaussianField:
            return K1
        if K0.is_GaussianRing:
            if K1.is_RationalField:
                K0 = K0.get_field()
            return K0
        if K1.is_GaussianRing:
            if K0.is_RationalField:
                K1 = K1.get_field()
            return K1
        if K0.is_RationalField:
            return K0
        if K1.is_RationalField:
            return K1
        if K0.is_IntegerRing:
            return K0
        if K1.is_IntegerRing:
            return K1
        from sympy.polys.domains import EX
        return EX
    n = evalf
