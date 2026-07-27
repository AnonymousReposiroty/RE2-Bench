from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
from sympy.core.numbers import I, pi, Rational

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

    def _eval_is_nonnegative(self):
        z = self.args[0]
        x, y = z.as_real_imag()
        ymod = y % (2 * pi)
        yzero = ymod.is_zero
        if yzero:
            return True
        xzero = x.is_zero
        if xzero is False:
            return yzero
        return fuzzy_or([yzero, fuzzy_and([xzero, fuzzy_or([ymod <= pi / 2, ymod >= 3 * pi / 2])])])
