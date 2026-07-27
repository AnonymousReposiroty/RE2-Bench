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

def dup_diff(f: dup[Er], m: int, K: Domain[Er]) -> dup[Er]:
    """
    ``m``-th order derivative of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 1)
    3*x**2 + 4*x + 3
    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 2)
    6*x + 4

    """
    if m <= 0:
        return f

    n = dup_degree(f)

    if n < m:
        return []

    deriv: list[Er] = []

    if m == 1:
        for coeff in f[:-m]:
            deriv.append(K(n)*coeff)
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n

            for i in range(n - 1, n - m, -1):
                k *= i

            deriv.append(K(k)*coeff)
            n -= 1

    return dup_strip(deriv, K)
