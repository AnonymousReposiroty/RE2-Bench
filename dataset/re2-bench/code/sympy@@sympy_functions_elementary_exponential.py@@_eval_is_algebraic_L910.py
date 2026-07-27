from sympy.core.expr import Expr
from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.singleton import S

class log(DefinedFunction):
    """
    The natural logarithm function `\\ln(x)` or `\\log(x)`.

    Explanation
    ===========

    Logarithms are taken with the natural base, `e`. To get
    a logarithm of a different base ``b``, use ``log(x, b)``,
    which is essentially short-hand for ``log(x)/log(b)``.

    ``log`` represents the principal branch of the natural
    logarithm. As such it has a branch cut along the negative
    real axis and returns values having a complex argument in
    `(-\\pi, \\pi]`.

    Examples
    ========

    >>> from sympy import log, sqrt, S, I
    >>> log(8, 2)
    3
    >>> log(S(8)/3, 2)
    -log(3)/log(2) + 3
    >>> log(-1 + I*sqrt(3))
    log(2) + 2*I*pi/3

    See Also
    ========

    sympy.functions.elementary.exponential.exp

    """
    args: tuple[Expr]
    _singularities = (S.Zero, S.ComplexInfinity)

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic
