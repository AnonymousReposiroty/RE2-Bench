from sympy.core.add import Add
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.singleton import S

def _peeloff_pi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $\pi$.
    This assumes ARG to be an Add.
    The multiple of $\pi$ returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    """
    pi_coeff = S.Zero
    rest_terms = []
    for a in Add.make_args(arg):
        K = a.coeff(pi)
        if K and K.is_rational:
            pi_coeff += K
        else:
            rest_terms.append(a)

    if pi_coeff is S.Zero:
        return arg, S.Zero

    m1 = (pi_coeff % S.Half)
    m2 = pi_coeff - m1
    if m2.is_integer or ((2*m2).is_integer and m2.is_even is False):
        return Add(*(rest_terms + [m1*pi])), m2
    return arg, S.Zero
