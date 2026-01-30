from sympy.polys.densebasic import (
    dup, dmp, _dup, _dmp,
    dup_strip,
    dup_LC, dmp_ground_LC,
    dmp_zero_p,
    dmp_ground,
    dup_degree, dmp_degree, dmp_degree_in, dmp_degree_list,
    dmp_raise, dmp_inject,
    dup_convert)
from sympy.polys.densetools import (
    dup_diff, dmp_diff, dmp_diff_in,
    dup_shift, dmp_shift,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive)
from sympy.polys.euclidtools import (
    dup_inner_gcd, dmp_inner_gcd,
    dup_gcd, dmp_gcd,
    dmp_resultant, dmp_primitive)

def dup_sqf_p(f, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    """
    if not f:
        return True
    else:
        return not dup_degree(dup_gcd(f, dup_diff(f, 1, K), K))
