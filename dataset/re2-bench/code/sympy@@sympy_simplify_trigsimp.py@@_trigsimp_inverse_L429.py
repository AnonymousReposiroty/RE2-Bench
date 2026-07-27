from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
                                 expand_mul, _coeff_isneg, Derivative)
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.trigonometric import TrigonometricFunction

def _trigsimp_inverse(rv):

    def check_args(x, y):
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    def f(rv):
        # for simple functions
        g = getattr(rv, 'inverse', None)
        if (g is not None and isinstance(rv.args[0], g()) and
                isinstance(g()(1), TrigonometricFunction)):
            return rv.args[0].args[0]

        # for atan2 simplifications, harder because atan2 has 2 args
        if isinstance(rv, atan2):
            y, x = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))

            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]

        return rv

    return bottom_up(rv, f)
