from sympy.core.singleton import S
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def TR11(rv, base=None):
    """Function of double angle to product. The ``base`` argument can be used
    to indicate what is the un-doubled argument, e.g. if 3*pi/7 is the base
    then cosine and sine functions with argument 6*pi/7 will be replaced.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11
    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import x
    >>> TR11(sin(2*x))
    2*sin(x)*cos(x)
    >>> TR11(cos(2*x))
    -sin(x)**2 + cos(x)**2
    >>> TR11(sin(4*x))
    4*(-sin(x)**2 + cos(x)**2)*sin(x)*cos(x)
    >>> TR11(sin(4*x/3))
    4*(-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3)

    If the arguments are simply integers, no change is made
    unless a base is provided:

    >>> TR11(cos(2))
    cos(2)
    >>> TR11(cos(4), 2)
    -sin(2)**2 + cos(2)**2

    There is a subtle issue here in that autosimplification will convert
    some higher angles to lower angles

    >>> cos(6*pi/7) + cos(3*pi/7)
    -cos(pi/7) + cos(3*pi/7)

    The 6*pi/7 angle is now pi/7 but can be targeted with TR11 by supplying
    the 3*pi/7 base:

    >>> TR11(_, 3*pi/7)
    -sin(3*pi/7)**2 + cos(3*pi/7)**2 + cos(3*pi/7)

    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        if base:
            f = rv.func
            t = f(base*2)
            co = S.One
            if t.is_Mul:
                co, t = t.as_coeff_Mul()
            if t.func not in (cos, sin):
                return rv
            if rv.args[0] == t.args[0]:
                c = cos(base)
                s = sin(base)
                if f is cos:
                    return (c**2 - s**2)/co
                else:
                    return 2*c*s/co
            return rv

        elif not rv.args[0].is_Number:
            # make a change if the leading coefficient's numerator is
            # divisible by 2
            c, m = rv.args[0].as_coeff_Mul(rational=True)
            if c.p % 2 == 0:
                arg = c.p//2*m/c.q
                c = TR11(cos(arg))
                s = TR11(sin(arg))
                if rv.func == sin:
                    rv = 2*s*c
                else:
                    rv = c**2 - s**2
        return rv

    return bottom_up(rv, f)
