from .kind import Kind, UndefinedKind
from ._print_helpers import Printable
from typing import ClassVar, TypeVar, Any, Hashable
from .assumptions import StdFactKB

class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """
    __slots__ = ('_mhash', '_args', '_assumptions')
    _args: tuple[Basic, ...]
    _mhash: int | None
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False
    default_assumptions: ClassVar[StdFactKB]
    is_composite: bool | None
    is_noninteger: bool | None
    is_extended_positive: bool | None
    is_negative: bool | None
    is_complex: bool | None
    is_extended_nonpositive: bool | None
    is_integer: bool | None
    is_positive: bool | None
    is_rational: bool | None
    is_extended_nonnegative: bool | None
    is_infinite: bool | None
    is_extended_negative: bool | None
    is_extended_real: bool | None
    is_finite: bool | None
    is_polar: bool | None
    is_imaginary: bool | None
    is_transcendental: bool | None
    is_extended_nonzero: bool | None
    is_nonzero: bool | None
    is_odd: bool | None
    is_algebraic: bool | None
    is_prime: bool | None
    is_commutative: bool | None
    is_nonnegative: bool | None
    is_nonpositive: bool | None
    is_irrational: bool | None
    is_real: bool | None
    is_zero: bool | None
    is_even: bool | None
    kind: Kind = UndefinedKind

    def _hashable_content(self) -> tuple[Hashable, ...]:
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        c = _cmp_name(n1, n2)
        if c:
            return c
        st = self._hashable_content()
        ot = other._hashable_content()
        len_st = len(st)
        len_ot = len(ot)
        c = (len_st > len_ot) - (len_st < len_ot)
        if c:
            return c
        for l, r in zip(st, ot):
            if isinstance(l, Basic):
                c = l.compare(r)
            elif isinstance(l, frozenset):
                l = Basic(*l) if isinstance(l, frozenset) else l
                r = Basic(*r) if isinstance(r, frozenset) else r
                c = l.compare(r)
            else:
                c = (l > r) - (l < r)
            if c:
                return c
        return 0
    _constructor_postprocessor_mapping = {}
