from .basic import Atom, Basic
from .numbers import int_valued
from .singleton import S
from .sympify import _sympify, SympifyError
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from .expr import Expr
from sympy.multipledispatch import dispatch
from sympy.assumptions.wrapper import AssumptionsWrapper, is_extended_nonnegative
from sympy.assumptions.wrapper import (AssumptionsWrapper,
    is_infinite, is_extended_real)
from .add import Add
from .add import Add
from .expr import Expr
from .add import _unevaluated_Add, Add
from .expr import Expr
from sympy.functions.elementary.complexes import arg
from .add import Add
from sympy.simplify.simplify import clear_coefficients

def is_eq(lhs: Basic, rhs: Basic, assumptions=None) -> bool | None:
    """
    Fuzzy bool representing mathematical equality between *lhs* and *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified.

    rhs : Expr
        The right-hand side of the expression, must be sympified.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the equality.

    Returns
    =======

    ``True`` if *lhs* is equal to *rhs*, ``False`` is *lhs* is not equal to *rhs*,
    and ``None`` if the comparison between *lhs* and *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    :func:`~.is_neq` calls this function to return its value, so supporting
    new type with this function will ensure correct behavior for ``is_neq``
    as well.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.core.relational import is_eq, is_neq
    >>> from sympy.abc import x
    >>> is_eq(S(0), S(0))
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> is_neq(S(0), S(2))
    True

    Assumptions can be passed to evaluate the equality which is otherwise
    indeterminate.

    >>> print(is_eq(x, S(0)))
    None
    >>> is_eq(x, S(0), assumptions=Q.zero(x))
    True

    New types can be supported by dispatching to ``_eval_is_eq``.

    >>> from sympy import Basic, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> is_eq(a, b)
    True
    >>> is_neq(a, b)
    False

    """
    # here, _eval_Eq is only called for backwards compatibility
    # new code should use is_eq with multiple dispatch as
    # outlined in the docstring
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval

    # retval is still None, so go through the equality logic
    # If expressions have the same structure, they must be equal.
    if lhs == rhs:
        return True  # e.g. True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # only Booleans can equal Booleans

    from sympy.assumptions.wrapper import (AssumptionsWrapper,
        is_infinite, is_extended_real)
    from .add import Add

    _lhs = AssumptionsWrapper(lhs, assumptions)
    _rhs = AssumptionsWrapper(rhs, assumptions)

    if _lhs.is_infinite or _rhs.is_infinite:
        if fuzzy_xor([_lhs.is_infinite, _rhs.is_infinite]):
            return False
        if fuzzy_xor([_lhs.is_extended_real, _rhs.is_extended_real]):
            return False
        if fuzzy_and([_lhs.is_extended_real, _rhs.is_extended_real]):
            return fuzzy_xor([_lhs.is_extended_positive, fuzzy_not(_rhs.is_extended_positive)])

        # Try to split real/imaginary parts and equate them
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            real_imag = lambda t: (
                'real' if is_extended_real(t, assumptions) else
                'imag' if is_extended_real(I*t, assumptions) else None)
            return sift(Add.make_args(expr), real_imag)

        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                eq_real = is_eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']), assumptions)
                eq_imag = is_eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']), assumptions)
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        from sympy.functions.elementary.complexes import arg
        # Compare e.g. zoo with 1+I*oo by comparing args
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # Guard against Eq(nan, nan) -> False
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(is_eq(arglhs, argrhs, assumptions))

    if isinstance(lhs, Expr) and isinstance(rhs, Expr):
        # see if the difference evaluates
        dif = lhs - rhs
        _dif = AssumptionsWrapper(dif, assumptions)
        z = _dif.is_zero
        if z is not None:
            if z is False and _dif.is_commutative:  # issue 10728
                return False
            if z:
                return True

        # is_zero cannot help decide integer/rational with Float
        c, t = dif.as_coeff_Add()
        if c.is_Float:
            if int_valued(c):
                if t.is_integer is False:
                    return False
            elif t.is_rational is False:
                return False

        n2 = _n2(lhs, rhs)
        if n2 is not None:
            return _sympify(n2 == 0)

        # see if the ratio evaluates
        n, d = dif.as_numer_denom()
        rv = None
        _n = AssumptionsWrapper(n, assumptions)
        _d = AssumptionsWrapper(d, assumptions)
        if _n.is_zero:
            rv = _d.is_nonzero
        elif _n.is_finite:
            if _d.is_infinite:
                rv = True
            elif _n.is_zero is False:
                rv = _d.is_infinite
                if rv is None:
                    # if the condition that makes the denominator
                    # infinite does not make the original expression
                    # True then False can be returned
                    from sympy.simplify.simplify import clear_coefficients
                    l, r = clear_coefficients(d, S.Infinity)
                    lhs2 = lhs.subs(l, r)
                    rhs2 = rhs.subs(l, r)
                    if lhs2 != lhs or rhs2 != rhs:
                        rv = fuzzy_bool(is_eq(lhs2, rhs2, assumptions))
                        if rv is True:
                            rv = None
        elif any(is_infinite(a, assumptions) for a in Add.make_args(n)):
            # (inf or nan)/x != 0
            rv = False
        if rv is not None:
            return rv

    return None
