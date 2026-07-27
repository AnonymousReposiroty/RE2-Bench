from sympy.polys.domains.domain import Domain, Er, Ef, Eeuclid, Eabs, Eordered
from sympy.polys.densebasic import (
    dup, dmp, _dup, _dmp, _dmp_ground,
    dup_slice, dup_truncate,
    dup_reverse,
    dup_LC, dmp_LC,
    dup_degree, dmp_degree,
    dup_strip, dmp_strip,
    dmp_zero_p, dmp_zero,
    dmp_one_p, dmp_one,
    dmp_ground, dmp_zeros)

def dup_sub_mul(f: dup[Er], g: dup[Er], h: dup[Er], K: Domain[Er]) -> dup[Er]:
    """
    Returns ``f - g*h`` where ``f, g, h`` are in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_mul(x**2 - 1, x - 2, x + 2)
    3

    """
    return dup_sub(f, dup_mul(g, h, K), K)
