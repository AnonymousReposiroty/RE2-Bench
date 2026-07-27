from sympy.core import S, sympify, cacheit
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, csc, sec, sin, tan,
    _imaginary_unit_as_coefficient)
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.elementary.trigonometric import (cos, sin)

class cosh(HyperbolicFunction):
    """
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sympy.functions.elementary.hyperbolic.sinh
    sympy.functions.elementary.hyperbolic.tanh
    sympy.functions.elementary.hyperbolic.acosh
    """

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (cosh(re) * cos(im), sinh(re) * sin(im))
