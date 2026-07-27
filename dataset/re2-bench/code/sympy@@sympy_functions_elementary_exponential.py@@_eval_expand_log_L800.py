from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.concrete import Sum, Product
from sympy.simplify.simplify import expand_log, simplify, inversecombine

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

    def _eval_expand_log(self, deep=True, **hints):
        from sympy.concrete import Sum, Product
        force = hints.get('force', False)
        factor = hints.get('factor', False)
        if len(self.args) == 2:
            return expand_log(self.func(*self.args), deep=deep, force=force)
        arg = self.args[0]
        if arg.is_Integer:
            p = perfect_power(arg)
            logarg = None
            coeff = 1
            if p is not False:
                arg, coeff = p
                logarg = self.func(arg)
            if factor:
                p = factorint(arg)
                if arg not in p.keys():
                    logarg = sum((n * log(val) for val, n in p.items()))
            if logarg is not None:
                return coeff * logarg
        elif arg.is_Rational:
            return log(arg.p) - log(arg.q)
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            return Add(*expr) + log(Mul(*nonpos))
        elif arg.is_Pow or isinstance(arg, exp):
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp + 1).is_positive and (arg.exp - 1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        elif isinstance(arg, Product):
            if force or arg.function.is_positive:
                return Sum(log(arg.function), *arg.limits)
        return self.func(arg)
