from .mul import Mul
from .sympify import sympify, _sympify
from sympy.functions.elementary.exponential import log
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction
from sympy import multiplicity

def expand_log(expr, deep=True, force=False, factor=False):
    """
    Wrapper around expand that only uses the log hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_log, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_log(exp(x+y)*(x+y)*log(x*y**2))
    (x + y)*(log(x) + 2*log(y))*exp(x + y)

    """
    from sympy.functions.elementary.exponential import log
    from sympy.simplify.radsimp import fraction
    if factor is False:
        def _handleMul(x):
            # look for the simple case of expanded log(b**a)/log(b) -> a in args
            n, d = fraction(x)
            n = [i for i in n.atoms(log) if i.args[0].is_Integer]
            d = [i for i in d.atoms(log) if i.args[0].is_Integer]
            if len(n) == 1 and len(d) == 1:
                n = n[0]
                d = d[0]
                from sympy import multiplicity
                m = multiplicity(d.args[0], n.args[0])
                if m:
                    r = m + log(n.args[0]//d.args[0]**m)/d
                    x = x.subs(n, d*r)
            x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
            if x1.count(log) <= x.count(log):
                return x1
            return x

        expr = expr.replace(
        lambda x: x.is_Mul and all(any(isinstance(i, log) and i.args[0].is_Rational
        for i in Mul.make_args(j)) for j in x.as_numer_denom()),
        _handleMul)

    return sympify(expr).expand(deep=deep, log=True, mul=False,
        power_exp=False, power_base=False, multinomial=False,
        basic=False, force=force, factor=factor)
