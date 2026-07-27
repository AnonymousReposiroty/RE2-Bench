from sympy.polys.domains.domain import Domain, Er, Eg, Ef, Eeuclid
from sympy.polys.domains.field import Field
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

def dmp_integrate_in(f: dmp[Ef], m: int, j: int, u: int, K: Field[Ef]) -> dmp[Ef]:
    """
    Computes the indefinite integral of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate_in(x + 2*y, 1, 0)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate_in(x + 2*y, 1, 1)
    x*y + y**2

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= u expected, got u = %d, j = %d" % (u, j))

    return _rec_integrate_in(f, m, u, 0, j, K)
