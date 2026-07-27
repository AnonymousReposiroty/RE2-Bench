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

def dmp_neg(f: dmp[Er], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Negate a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_neg(x**2*y - x)
    -x**2*y + x

    """
    if not u:
        return _dmp(dup_neg(_dup(f), K))

    v = u - 1

    return [ dmp_neg(cf, v, K) for cf in f ]
