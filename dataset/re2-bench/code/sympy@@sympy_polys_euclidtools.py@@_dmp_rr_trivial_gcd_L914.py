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
from sympy.polys.polyconfig import query

def _dmp_rr_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    if_contain_one = dmp_one_p(f, u, K) or dmp_one_p(g, u, K)

    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        if K.is_nonnegative(dmp_ground_LC(g, u, K)):
            return g, dmp_zero(u, K), dmp_one(u, K)
        else:
            return dmp_neg(g, u, K), dmp_zero(u, K), dmp_ground(-K.one, u, K)
    elif zero_g:
        if K.is_nonnegative(dmp_ground_LC(f, u, K)):
            return f, dmp_one(u, K), dmp_zero(u, K)
        else:
            return dmp_neg(f, u, K), dmp_ground(-K.one, u, K), dmp_zero(u, K)
    elif if_contain_one:
        return dmp_one(u, K), f, g
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None
