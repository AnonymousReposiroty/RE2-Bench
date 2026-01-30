from .singleton import S
from mpmath.libmp import mpf_log, prec_to_dps
from .numbers import Float, Integer, Rational, _illegal, int_valued
from math import log10, ceil, log
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.exponential import log

def _mag(x):
    r"""Return integer $i$ such that $0.1 \le x/10^i < 1$

    Examples
    ========

    >>> from sympy.core.expr import _mag
    >>> from sympy import Float
    >>> _mag(Float(.1))
    0
    >>> _mag(Float(.01))
    -1
    >>> _mag(Float(1234))
    4
    """
    from math import log10, ceil, log
    xpos = abs(x.n())
    if not xpos:
        return S.Zero
    try:
        mag_first_dig = int(ceil(log10(xpos)))
    except (ValueError, OverflowError):
        mag_first_dig = int(ceil(Float(mpf_log(xpos._mpf_, 53))/log(10)))
    # check that we aren't off by 1
    if (xpos/S(10)**mag_first_dig) >= 1:
        assert 1 <= (xpos/S(10)**mag_first_dig) < 10
        mag_first_dig += 1
    return mag_first_dig
