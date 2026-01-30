from sympy.core.power import Pow
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def TR15(rv, max=4, pow=False):
    """Convert sin(x)**-2 to 1 + cot(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR15
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR15(1 - 1/sin(x)**2)
    -cot(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, sin)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, sin, cot, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    return bottom_up(rv, f)
