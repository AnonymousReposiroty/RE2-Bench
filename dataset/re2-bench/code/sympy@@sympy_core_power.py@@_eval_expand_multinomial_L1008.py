from typing import Callable, TYPE_CHECKING
from .singleton import S
from .expr import Expr
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)
from .add import Add
from .numbers import Integer, Rational
from .mul import Mul, _keep_coeff
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.polyutils import basic_from_dict
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.exponential import exp, log

class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible than floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    """
    is_Pow = True
    __slots__ = ('is_commutative',)
    if TYPE_CHECKING:

        @property
        def args(self) -> tuple[Expr, Expr]:
            ...

    def _eval_expand_multinomial(self, **hints):
        """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""
        base, exp = self.args
        result = self
        if exp.is_Rational and exp.p > 0 and base.is_Add:
            if not exp.is_Integer:
                n = Integer(exp.p // exp.q)
                if not n:
                    return result
                else:
                    radical, result = (self.func(base, exp - n), [])
                    expanded_base_n = self.func(base, n)
                    if expanded_base_n.is_Pow:
                        expanded_base_n = expanded_base_n._eval_expand_multinomial()
                    for term in Add.make_args(expanded_base_n):
                        result.append(term * radical)
                    return Add(*result)
            n = int(exp)
            if base.is_commutative:
                order_terms, other_terms = ([], [])
                for b in base.args:
                    if b.is_Order:
                        order_terms.append(b)
                    else:
                        other_terms.append(b)
                if order_terms:
                    f = Add(*other_terms)
                    o = Add(*order_terms)
                    if n == 2:
                        return expand_multinomial(f ** n, deep=False) + n * f * o
                    else:
                        g = expand_multinomial(f ** (n - 1), deep=False)
                        return expand_mul(f * g, deep=False) + n * g * o
                if base.is_number:
                    a, b = base.as_real_imag()
                    if a.is_Rational and b.is_Rational:
                        if not a.is_Integer:
                            if not b.is_Integer:
                                k = self.func(a.q * b.q, n)
                                a, b = (a.p * b.q, a.q * b.p)
                            else:
                                k = self.func(a.q, n)
                                a, b = (a.p, a.q * b)
                        elif not b.is_Integer:
                            k = self.func(b.q, n)
                            a, b = (a * b.q, b.p)
                        else:
                            k = 1
                        a, b, c, d = (int(a), int(b), 1, 0)
                        while n:
                            if n & 1:
                                c, d = (a * c - b * d, b * c + a * d)
                                n -= 1
                            a, b = (a * a - b * b, 2 * a * b)
                            n //= 2
                        I = S.ImaginaryUnit
                        if k == 1:
                            return c + I * d
                        else:
                            return Integer(c) / k + I * d / k
                p = other_terms
                from sympy.ntheory.multinomial import multinomial_coefficients
                from sympy.polys.polyutils import basic_from_dict
                expansion_dict = multinomial_coefficients(len(p), n)
                return basic_from_dict(expansion_dict, *p)
            elif n == 2:
                return Add(*[f * g for f in base.args for g in base.args])
            else:
                multi = (base ** (n - 1))._eval_expand_multinomial()
                if multi.is_Add:
                    return Add(*[f * g for f in base.args for g in multi.args])
                else:
                    return Add(*[f * multi for f in base.args])
        elif exp.is_Rational and exp.p < 0 and base.is_Add and (abs(exp.p) > exp.q):
            return 1 / self.func(base, -exp)._eval_expand_multinomial()
        elif exp.is_Add and base.is_Number and (hints.get('force', False) or base.is_zero is False or exp._all_nonneg_or_nonppos()):
            coeff, tail = ([], [])
            for term in exp.args:
                if term.is_Number:
                    coeff.append(self.func(base, term))
                else:
                    tail.append(term)
            return Mul(*coeff + [self.func(base, Add._from_args(tail))])
        else:
            return result
