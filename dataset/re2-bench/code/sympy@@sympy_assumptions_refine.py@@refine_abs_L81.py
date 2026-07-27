from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.assumptions import ask, Q  # type: ignore
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.complexes import Abs

def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    from sympy.functions.elementary.complexes import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # arg is Mul
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))
