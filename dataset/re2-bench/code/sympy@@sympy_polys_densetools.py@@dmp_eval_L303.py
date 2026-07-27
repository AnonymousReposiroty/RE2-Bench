from sympy.polys.domains.domain import Domain, Er, Eg, Ef, Eeuclid
from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_lshift, dup_rshift,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul, dup_series_mul,
    dup_sqr,
    dup_div,
    dup_series_pow,
    dup_rem, dmp_rem,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
)
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

def dmp_eval(f: dmp[Er], a: Er, u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Evaluate a polynomial at ``x_0 = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_eval(2*x*y + 3*x + y + 2, 2)
    5*y + 8

    """
    if not u:
        return _ground_dmp(dup_eval(_dup(f), a, K))

    if not a:
        return dmp_TC(f, K)

    result, v = dmp_LC(f, K), u - 1

    for coeff in f[1:]:
        result = dmp_mul_ground(result, a, v, K)
        result = dmp_add(result, coeff, v, K)

    return result
