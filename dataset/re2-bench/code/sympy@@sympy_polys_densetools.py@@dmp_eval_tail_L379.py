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

def dmp_eval_tail(f: dmp[Er], A: list[Er], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Evaluate a polynomial at ``x_j = a_j, ...`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_tail(f, [2])
    7*x + 4
    >>> R.dmp_eval_tail(f, [2, 2])
    18

    """
    if not A:
        return f

    if dmp_zero_p(f, u):
        return dmp_zero(u - len(A), K)

    e = _rec_eval_tail(f, 0, A, u, K)

    if u == len(A) - 1:
        return e
    else:
        return dmp_strip(e, u - len(A), K)
