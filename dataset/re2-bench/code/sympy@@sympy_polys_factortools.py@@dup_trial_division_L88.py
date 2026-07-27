from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr,
    dmp_pow,
    dup_div, dmp_div,
    dup_quo, dmp_quo,
    dmp_expand,
    dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_lshift,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground)
from sympy.polys.polyutils import _sort_factors

def dup_trial_division(f, factors, K):
    """
    Determine multiplicities of factors for a univariate polynomial
    using trial division.

    An error will be raised if any factor does not divide ``f``.
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dup_div(f, factor, K)

            if not r:
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)
