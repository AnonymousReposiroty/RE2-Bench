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

def dmp_max_norm(f: dmp[Eordered], u: int, K: Domain[Eordered]) -> Eordered:
    """
    Returns maximum norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_max_norm(2*x*y - x - 3)
    3

    """
    if not u:
        return dup_max_norm(_dup(f), K)

    v = u - 1

    return max(dmp_max_norm(c, v, K) for c in f)
