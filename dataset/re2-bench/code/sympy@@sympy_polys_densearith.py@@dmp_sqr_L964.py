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

def dmp_sqr(f: dmp[Er], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Square dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqr(x**2 + x*y + y**2)
    x**4 + 2*x**3*y + 3*x**2*y**2 + 2*x*y**3 + y**4

    """
    if not u:
        return _dmp(dup_sqr(_dup(f), K))

    df = dmp_degree(f, u)

    if df < 0:
        return f

    h: list[dmp[Er]] = []
    v = u - 1

    for i in range(0, 2*df + 1):
        c = dmp_zero(v, K)

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            c = dmp_add(c, dmp_mul(f[j], f[i - j], v, K), v, K)

        c = dmp_mul_ground(c, K(2), v, K)

        if n & 1:
            elem = dmp_sqr(f[jmax + 1], v, K)
            c = dmp_add(c, elem, v, K)

        h.append(c)

    return dmp_strip(h, u, K)
