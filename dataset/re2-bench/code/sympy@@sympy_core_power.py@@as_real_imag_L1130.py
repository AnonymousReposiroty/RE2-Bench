from typing import Callable, TYPE_CHECKING
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)
from .add import Add
from .symbol import Symbol, Dummy, symbols
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.trigonometric import atan2, cos, sin
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.exponential import exp, log
from .sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.exponential import log
from sympy.polys.polytools import poly
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.complexes import arg, im, re, sign
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import log
from sympy.simplify.radsimp import fraction
from sympy.functions.elementary.complexes import sign, im
from sympy.functions.elementary.exponential import log

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

    @property
    def base(self) -> Expr:
        return self.args[0]

    @property
    def exp(self) -> Expr:
        return self.args[1]

    def as_real_imag(self, deep=True, **hints):
        if self.exp.is_Integer:
            from sympy.polys.polytools import poly
            exp = self.exp
            re_e, im_e = self.base.as_real_imag(deep=deep)
            if not im_e:
                return (self, S.Zero)
            a, b = symbols('a b', cls=Dummy)
            if exp >= 0:
                if re_e.is_Number and im_e.is_Number:
                    expr = expand_multinomial(self.base ** exp)
                    if expr != self:
                        return expr.as_real_imag()
                expr = poly((a + b) ** exp)
            else:
                mag = re_e ** 2 + im_e ** 2
                re_e, im_e = (re_e / mag, -im_e / mag)
                if re_e.is_Number and im_e.is_Number:
                    expr = expand_multinomial((re_e + im_e * S.ImaginaryUnit) ** (-exp))
                    if expr != self:
                        return expr.as_real_imag()
                expr = poly((a + b) ** (-exp))
            r = [i for i in expr.terms() if not i[0][1] % 2]
            re_part = Add(*[cc * a ** aa * b ** bb for (aa, bb), cc in r])
            r = [i for i in expr.terms() if i[0][1] % 4 == 1]
            im_part1 = Add(*[cc * a ** aa * b ** bb for (aa, bb), cc in r])
            r = [i for i in expr.terms() if i[0][1] % 4 == 3]
            im_part3 = Add(*[cc * a ** aa * b ** bb for (aa, bb), cc in r])
            return (re_part.subs({a: re_e, b: S.ImaginaryUnit * im_e}), im_part1.subs({a: re_e, b: im_e}) + im_part3.subs({a: re_e, b: -im_e}))
        from sympy.functions.elementary.trigonometric import atan2, cos, sin
        if self.exp.is_Rational:
            re_e, im_e = self.base.as_real_imag(deep=deep)
            if im_e.is_zero and self.exp is S.Half:
                if re_e.is_extended_nonnegative:
                    return (self, S.Zero)
                if re_e.is_extended_nonpositive:
                    return (S.Zero, (-self.base) ** self.exp)
            r = self.func(self.func(re_e, 2) + self.func(im_e, 2), S.Half)
            t = atan2(im_e, re_e)
            rp, tp = (self.func(r, self.exp), t * self.exp)
            return (rp * cos(tp), rp * sin(tp))
        elif self.base is S.Exp1:
            from sympy.functions.elementary.exponential import exp
            re_e, im_e = self.exp.as_real_imag()
            if deep:
                re_e = re_e.expand(deep, **hints)
                im_e = im_e.expand(deep, **hints)
            c, s = (cos(im_e), sin(im_e))
            return (exp(re_e) * c, exp(re_e) * s)
        else:
            from sympy.functions.elementary.complexes import im, re
            if deep:
                hints['complex'] = False
                expanded = self.expand(deep, **hints)
                if hints.get('ignore') == expanded:
                    return None
                else:
                    return (re(expanded), im(expanded))
            else:
                return (re(self), im(self))

    @cacheit
    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True, mul=True, log=True, multinomial=True, basic=True, **hints):
        """
        Expand an expression using hints.

        See the docstring of the expand() function in sympy.core.function for
        more information.

        """
        from sympy.simplify.radsimp import fraction
        hints.update(power_base=power_base, power_exp=power_exp, mul=mul, log=log, multinomial=multinomial, basic=basic)
        expr = self
        _fraction = lambda x: fraction(x, hints.get('exact', False))
        if hints.pop('frac', False):
            n, d = [a.expand(deep=deep, modulus=modulus, **hints) for a in _fraction(self)]
            return n / d
        elif hints.pop('denom', False):
            n, d = _fraction(self)
            return n / d.expand(deep=deep, modulus=modulus, **hints)
        elif hints.pop('numer', False):
            n, d = _fraction(self)
            return n.expand(deep=deep, modulus=modulus, **hints) / d

        def _expand_hint_key(hint):
            """Make multinomial come before mul"""
            if hint == 'mul':
                return 'mulz'
            return hint
        for hint in sorted(hints.keys(), key=_expand_hint_key):
            use_hint = hints[hint]
            if use_hint:
                hint = '_eval_expand_' + hint
                expr, hit = Expr._expand_hint(expr, hint, deep=deep, **hints)
        while True:
            was = expr
            if hints.get('multinomial', False):
                expr, _ = Expr._expand_hint(expr, '_eval_expand_multinomial', deep=deep, **hints)
            if hints.get('mul', False):
                expr, _ = Expr._expand_hint(expr, '_eval_expand_mul', deep=deep, **hints)
            if hints.get('log', False):
                expr, _ = Expr._expand_hint(expr, '_eval_expand_log', deep=deep, **hints)
            if expr == was:
                break
        if modulus is not None:
            modulus = sympify(modulus)
            if not modulus.is_Integer or modulus <= 0:
                raise ValueError('modulus must be a positive integer, got %s' % modulus)
            terms = []
            for term in Add.make_args(expr):
                coeff, tail = term.as_coeff_Mul(rational=True)
                coeff %= modulus
                if coeff:
                    terms.append(coeff * tail)
            expr = Add(*terms)
        return expr
