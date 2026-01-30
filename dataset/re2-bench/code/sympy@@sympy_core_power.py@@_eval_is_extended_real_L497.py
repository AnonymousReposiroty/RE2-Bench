from typing import Callable, TYPE_CHECKING
from .singleton import S
from .expr import Expr
from .numbers import Integer, Rational
from .mul import Mul, _keep_coeff
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.complexes import conjugate as c
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.complexes import arg
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.complexes import arg
from sympy.functions.elementary.complexes import arg, Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.complexes import arg, im, re, sign
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import log
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

    def _eval_is_extended_real(self):
        if self.base is S.Exp1:
            if self.exp.is_extended_real:
                return True
            elif self.exp.is_imaginary:
                return (2 * S.ImaginaryUnit * self.exp / S.Pi).is_even
        from sympy.functions.elementary.exponential import log, exp
        real_b = self.base.is_extended_real
        if real_b is None:
            if self.base.func == exp and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            if self.base.func == Pow and self.base.base is S.Exp1 and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_extended_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_extended_positive:
                return True
            elif self.base.is_extended_nonnegative and self.exp.is_extended_nonnegative:
                return True
            elif self.exp.is_integer and self.base.is_extended_nonzero:
                return True
            elif self.exp.is_integer and self.exp.is_nonnegative:
                return True
            elif self.base.is_extended_negative:
                if self.exp.is_Rational:
                    return False
        if real_e and self.exp.is_extended_negative and (self.base.is_zero is False):
            return Pow(self.base, -self.exp).is_extended_real
        im_b = self.base.is_imaginary
        im_e = self.exp.is_imaginary
        if im_b:
            if self.exp.is_integer:
                if self.exp.is_even:
                    return True
                elif self.exp.is_odd:
                    return False
            elif im_e and log(self.base).is_imaginary:
                return True
            elif self.exp.is_Add:
                c, a = self.exp.as_coeff_Add()
                if c and c.is_Integer:
                    return Mul(self.base ** c, self.base ** a, evaluate=False).is_extended_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp / 2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                if self.base.is_rational and c.is_rational:
                    if self.base.is_nonzero and (self.base - 1).is_nonzero and c.is_nonzero:
                        return False
                ok = (c * log(self.base) / S.Pi).is_integer
                if ok is not None:
                    return ok
        if real_b is False and real_e:
            if isinstance(self.exp, Rational) and self.exp.p == 1:
                return False
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base) * self.exp / S.Pi
            if i.is_complex:
                return i.is_integer
