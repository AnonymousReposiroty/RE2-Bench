from typing import Callable, TYPE_CHECKING
from .singleton import S
from .expr import Expr
from .numbers import Integer, Rational
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import exp, log
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

    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        If base a Rational less than 1, then return 1/Rational, -exp.
        If this extra processing is not needed, the base and exp
        properties will give the raw arguments.

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """
        b, e = self.args
        if b.is_Rational and b.p == 1 and (b.q != 1):
            return (Integer(b.q), -e)
        return (b, e)

    def as_numer_denom(self):
        if not self.is_commutative:
            return (self, S.One)
        base, exp = self.as_base_exp()
        n, d = base.as_numer_denom()
        neg_exp = exp.is_negative
        if exp.is_Mul and (not neg_exp) and (not exp.is_positive):
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer
        if not (d.is_extended_real or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive
        if dnonpos:
            n, d = (-n, -d)
        elif dnonpos is None and (not int_exp):
            n = base
            d = S.One
        if neg_exp:
            n, d = (d, n)
            exp = -exp
        if exp.is_infinite:
            if n is S.One and d is not S.One:
                return (n, self.func(d, exp))
            if n is not S.One and d is S.One:
                return (self.func(n, exp), d)
        return (self.func(n, exp), self.func(d, exp))
