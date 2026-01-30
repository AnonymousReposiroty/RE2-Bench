from .assumptions import StdFactKB, _assume_defined
from .expr import Expr, AtomicExpr
from sympy.logic.boolalg import Boolean
from sympy.functions.elementary.complexes import im, re

class Symbol(AtomicExpr, Boolean):
    """
    Symbol class is used to create symbolic variables.

    Explanation
    ===========

    Symbolic variables are placeholders for mathematical symbols that can represent numbers, constants, or any other mathematical entities and can be used in mathematical expressions and to perform symbolic computations.

    Assumptions:

    commutative = True
    positive = True
    real = True
    imaginary = True
    complex = True
    complete list of more assumptions- :ref:`predicates`

    You can override the default assumptions in the constructor.

    Examples
    ========

    >>> from sympy import Symbol
    >>> x = Symbol("x", positive=True)
    >>> x.is_positive
    True
    >>> x.is_negative
    False

    passing in greek letters:

    >>> from sympy import Symbol
    >>> alpha = Symbol('alpha')
    >>> alpha #doctest: +SKIP
    α

    Trailing digits are automatically treated like subscripts of what precedes them in the name.
    General format to add subscript to a symbol :
    ``<var_name> = Symbol('<symbol_name>_<subscript>')``

    >>> from sympy import Symbol
    >>> alpha_i = Symbol('alpha_i')
    >>> alpha_i #doctest: +SKIP
    αᵢ

    Parameters
    ==========

    AtomicExpr: variable name
    Boolean: Assumption with a boolean value(True or False)
    """
    is_comparable = False
    __slots__ = ('name', '_assumptions_orig', '_assumptions0')
    name: str
    _assumptions_orig: dict[str, bool | None]
    _assumptions0: tuple[tuple[str, bool | None], ...]
    _assumptions: StdFactKB
    is_Symbol = True
    is_symbol = True

    def as_real_imag(self, deep=True, **hints) -> tuple[Expr, Expr]:
        if hints.get('ignore') == self:
            return None
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))
    binary_symbols = free_symbols
