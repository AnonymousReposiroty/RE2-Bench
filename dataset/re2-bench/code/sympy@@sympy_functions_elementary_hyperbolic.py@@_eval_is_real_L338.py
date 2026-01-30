from sympy.core.numbers import I, pi, Rational
from sympy.functions.elementary.complexes import Abs, im, re

class sinh(HyperbolicFunction):
    """
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    sympy.functions.elementary.hyperbolic.cosh
    sympy.functions.elementary.hyperbolic.tanh
    sympy.functions.elementary.hyperbolic.asinh
    """

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True
        re, im = arg.as_real_imag()
        return (im % pi).is_zero
