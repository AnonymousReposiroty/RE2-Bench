from sympy.polys.domains.domain import Domain, Er, Eg, Ef, Eeuclid
from sympy.polys.densebasic import (
    dup, dmp, _dup, _dmp, _dmp2, _ground_dmp,
    dup_strip, dmp_strip, dup_truncate,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_to_dict,
    dmp_from_dict,
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC, dmp_TC,
    dmp_zero, dmp_ground,
    dmp_zero_p,
    dup_to_raw_dict, dup_from_raw_dict,
    dmp_to_raw_dict,
    dmp_zeros,
    dmp_include,
    dup_nth,
)
from sympy.polys.domains import QQ
from sympy.polys.domains import QQ

def dmp_ground_content(f: dmp[Er], u: int, K: Domain[Er]) -> Er:
    """
    Compute the GCD of coefficients of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    """
    from sympy.polys.domains import QQ

    if not u:
        return dup_content(_dup(f), K)

    if dmp_zero_p(f, u):
        return K.zero

    cont, v = K.zero, u - 1

    if K == QQ:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))
    else:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))

            if K.is_one(cont):
                break

    return cont
