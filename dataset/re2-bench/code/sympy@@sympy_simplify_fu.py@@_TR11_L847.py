from collections import defaultdict
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def _TR11(rv):
    """
    Helper for TR11 to find half-arguments for sin in factors of
    num/den that appear in cos or sin factors in the den/num.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11, _TR11
    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> TR11(sin(x/3)/(cos(x/6)))
    sin(x/3)/cos(x/6)
    >>> _TR11(sin(x/3)/(cos(x/6)))
    2*sin(x/6)
    >>> TR11(sin(x/6)/(sin(x/3)))
    sin(x/6)/sin(x/3)
    >>> _TR11(sin(x/6)/(sin(x/3)))
    1/(2*cos(x/6))

    """
    def f(rv):
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            # find arguments of sin and cos that
            # appears as bases in args of flat
            # and have Integer exponents
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                b, e = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        num_args, den_args = map(sincos_args, rv.as_numer_denom())
        def handle_match(rv, num_args, den_args):
            # for arg in sin args of num_args, look for arg/2
            # in den_args and pass this half-angle to TR11
            # for handling in rv
            for narg in num_args[sin]:
                half = narg/2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        # sin in num, sin or cos in den
        rv = handle_match(rv, num_args, den_args)
        # sin in den, sin or cos in num
        rv = handle_match(rv, den_args, num_args)
        return rv

    return bottom_up(rv, f)
