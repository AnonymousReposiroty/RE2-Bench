from sympy.core.numbers import Integer, Rational, pi, I
from sympy.core.singleton import S

class exp(ExpBase, metaclass=ExpMeta):
    """
    The exponential function, :math:`e^x`.

    Examples
    ========

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import x
    >>> exp(x)
    exp(x)
    >>> exp(x).diff(x)
    exp(x)
    >>> exp(I*pi)
    -1

    Parameters
    ==========

    arg : Expr

    See Also
    ========

    sympy.functions.elementary.exponential.log
    """

    def _eval_is_extended_positive(self):
        if self.exp.is_extended_real:
            return self.args[0] is not S.NegativeInfinity
        elif self.exp.is_imaginary:
            arg2 = -I * self.args[0] / pi
            return arg2.is_even
