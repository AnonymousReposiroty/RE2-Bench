from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.numbers import Integer, Rational, pi, I

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

    def _eval_is_algebraic(self):
        if (self.exp / pi / I).is_rational:
            return True
        if fuzzy_not(self.exp.is_zero):
            if self.exp.is_algebraic:
                return False
            elif (self.exp / pi).is_rational:
                return False
