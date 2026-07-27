from sympy.core.expr import Expr
from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs

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

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I, log
        >>> from sympy.abc import x
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        sarg = self.args[0]
        if deep:
            sarg = self.args[0].expand(deep, **hints)
        sarg_abs = Abs(sarg)
        if sarg_abs == sarg:
            return (self, S.Zero)
        sarg_arg = arg(sarg)
        if hints.get('log', False):
            hints['complex'] = False
            return (log(sarg_abs).expand(deep, **hints), sarg_arg)
        else:
            return (log(sarg_abs), sarg_arg)
