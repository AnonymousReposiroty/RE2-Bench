from sympy.polys.domains.domain import Domain, Er, Eg, Ef, Eeuclid
from sympy.polys.domains.field import Field
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

def dmp_integrate(f: dmp[Ef], m: int, u: int, K: Field[Ef]) -> dmp[Ef]:
    """
    Computes the indefinite integral of ``f`` in ``x_0`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate(x + 2*y, 1)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate(x + 2*y, 2)
    1/6*x**3 + x**2*y

    """
    if not u:
        return _dmp(dup_integrate(_dup(f), m, K))

    if m <= 0 or dmp_zero_p(f, u):
        return f

    g, v = dmp_zeros(m, u - 1, K), u - 1

    for i, c in enumerate(reversed(f)):
        n = i + 1

        for j in range(1, m):
            n *= i + j + 1

        g.insert(0, dmp_quo_ground(c, K(n), v, K))

    return g
