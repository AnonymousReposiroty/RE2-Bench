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

def _rec_diff_in(g: dmp[Er], m: int, v: int, i: int, j: int, K: Domain[Er]) -> dmp[Er]:
    """Recursive helper for :func:`dmp_diff_in`."""
    if i == j:
        return dmp_diff(g, m, v, K)

    w, i = v - 1, i + 1

    return dmp_strip([ _rec_diff_in(c, m, w, i, j, K) for c in g ], v, K)
