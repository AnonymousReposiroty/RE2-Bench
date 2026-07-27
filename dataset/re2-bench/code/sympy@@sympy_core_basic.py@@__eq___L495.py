from .sympify import _sympify, sympify, SympifyError, _external_converter
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

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True
        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)
        if not (self.is_Number and other.is_Number) and type(self) != type(other):
            return False
        a, b = (self._hashable_content(), other._hashable_content())
        if a != b:
            return False
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True
    _constructor_postprocessor_mapping = {}
