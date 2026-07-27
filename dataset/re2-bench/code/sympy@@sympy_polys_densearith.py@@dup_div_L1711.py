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

def dup_div(f: dup[Er], g: dup[Er], K: Domain[Er]) -> tuple[dup[Er], dup[Er]]:
    """
    Polynomial division with remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (0, x**2 + 1)

    >>> R, x = ring("x", QQ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (1/2*x + 1, 5)

    """
    if K.is_Field:
        return dup_ff_div(f, g, K) # type: ignore
    else:
        return dup_rr_div(f, g, K) # type: ignore
