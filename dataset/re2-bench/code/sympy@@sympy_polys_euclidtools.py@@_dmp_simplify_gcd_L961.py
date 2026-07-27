from sympy.polys.densearith import (
    dup_sub_mul,
    dup_neg, dmp_neg,
    dmp_add,
    dmp_sub,
    dup_mul, dmp_mul,
    dmp_pow,
    dup_div, dmp_div,
    dup_rem,
    dup_quo, dmp_quo,
    dup_prem, dmp_prem,
    dup_mul_ground, dmp_mul_ground,
    dmp_mul_term,
    dup_quo_ground, dmp_quo_ground,
    dup_max_norm, dmp_max_norm)
from sympy.polys.densebasic import (
    dup, dmp, _dup, _dmp,
    dup_strip, dmp_raise,
    dmp_zero, dmp_one, dmp_ground,
    dmp_one_p, dmp_zero_p,
    dmp_zeros,
    dup_degree, dmp_degree, dmp_degree_in,
    dup_LC, dmp_LC, dmp_ground_LC,
    dmp_multi_deflate, dmp_inflate,
    dup_convert, dmp_convert,
    dmp_apply_pairs)

def _dmp_simplify_gcd(f, g, u, K):
    """Try to eliminate `x_0` from GCD computation in `K[X]`. """
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if df > 0 and dg > 0:
        return None

    if not (df or dg):
        F = dmp_LC(f, K)
        G = dmp_LC(g, K)
    else:
        if not df:
            F = dmp_LC(f, K)
            G = dmp_content(g, u, K)
        else:
            F = dmp_content(f, u, K)
            G = dmp_LC(g, K)

    v = u - 1
    h = dmp_gcd(F, G, v, K)

    cff = [ dmp_quo(cf, h, v, K) for cf in f ]
    cfg = [ dmp_quo(cg, h, v, K) for cg in g ]

    return [h], cff, cfg
