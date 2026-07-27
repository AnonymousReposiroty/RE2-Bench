from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.simplify.fu import hyper_as_trig, TR2i
from sympy.simplify.fu import hyper_as_trig

def futrig(e, *, hyper=True, **kwargs):
    """Return simplified ``e`` using Fu-like transformations.
    This is not the "Fu" algorithm. This is called by default
    from ``trigsimp``. By default, hyperbolics subexpressions
    will be simplified, but this can be disabled by setting
    ``hyper=False``.

    Examples
    ========

    >>> from sympy import trigsimp, tan, sinh, tanh
    >>> from sympy.simplify.trigsimp import futrig
    >>> from sympy.abc import x
    >>> trigsimp(1/tan(x)**2)
    tan(x)**(-2)

    >>> futrig(sinh(x)/tanh(x))
    cosh(x)

    """
    from sympy.simplify.fu import hyper_as_trig

    e = sympify(e)

    if not isinstance(e, Basic):
        return e

    if not e.args:
        return e

    old = e
    e = bottom_up(e, _futrig)

    if hyper and e.has(HyperbolicFunction):
        e, f = hyper_as_trig(e)
        e = f(bottom_up(e, _futrig))

    if e != old and e.is_Mul and e.args[0].is_Rational:
        # redistribute leading coeff on 2-arg Add
        e = Mul(*e.as_coeff_Mul())
    return e
