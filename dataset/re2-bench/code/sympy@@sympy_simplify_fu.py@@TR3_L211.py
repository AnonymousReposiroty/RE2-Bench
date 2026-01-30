from sympy.core.singleton import S
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)
from sympy.simplify.simplify import signsimp
from sympy.simplify.simplify import signsimp

def TR3(rv):
    """Induced formula: example sin(-a) = -sin(a)

    Examples
    ========

    >>> from sympy.simplify.fu import TR3
    >>> from sympy.abc import x, y
    >>> from sympy import pi
    >>> from sympy import cos
    >>> TR3(cos(y - x*(y - x)))
    cos(x*(x - y) + y)
    >>> cos(pi/2 + x)
    -sin(x)
    >>> cos(30*pi/2 + x)
    -cos(x)

    """
    from sympy.simplify.simplify import signsimp

    # Negative argument (already automatic for funcs like sin(-x) -> -sin(x)
    # but more complicated expressions can use it, too). Also, trig angles
    # between pi/4 and pi/2 are not reduced to an angle between 0 and pi/4.
    # The following are automatically handled:
    #   Argument of type: pi/2 +/- angle
    #   Argument of type: pi +/- angle
    #   Argument of type : 2k*pi +/- angle

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        rv = rv.func(signsimp(rv.args[0]))
        if not isinstance(rv, TrigonometricFunction):
            return rv
        if (rv.args[0] - S.Pi/4).is_positive is (S.Pi/2 - rv.args[0]).is_positive is True:
            fmap = {cos: sin, sin: cos, tan: cot, cot: tan, sec: csc, csc: sec}
            rv = fmap[type(rv)](S.Pi/2 - rv.args[0])
        return rv

    # touch numbers iside of trig functions to let them automatically update
    rv = rv.replace(
        lambda x: isinstance(x, TrigonometricFunction),
        lambda x: x.replace(
            lambda n: n.is_number and n.is_Mul,
            lambda n: n.func(*n.args)))

    return bottom_up(rv, f)
