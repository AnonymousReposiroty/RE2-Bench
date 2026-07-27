from sympy.core.add import Add
from sympy.core.sorting import ordered
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def TR10(rv, first=True):
    """Separate sums in ``cos`` and ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10
    >>> from sympy.abc import a, b, c
    >>> from sympy import cos, sin
    >>> TR10(cos(a + b))
    -sin(a)*sin(b) + cos(a)*cos(b)
    >>> TR10(sin(a + b))
    sin(a)*cos(b) + sin(b)*cos(a)
    >>> TR10(sin(a + b + c))
    (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
    (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        f = rv.func
        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                if f == sin:
                    return sin(a)*TR10(cos(b), first=False) + \
                        cos(a)*TR10(sin(b), first=False)
                else:
                    return cos(a)*TR10(cos(b), first=False) - \
                        sin(a)*TR10(sin(b), first=False)
            else:
                if f == sin:
                    return sin(a)*cos(b) + cos(a)*sin(b)
                else:
                    return cos(a)*cos(b) - sin(a)*sin(b)
        return rv

    return bottom_up(rv, f)
