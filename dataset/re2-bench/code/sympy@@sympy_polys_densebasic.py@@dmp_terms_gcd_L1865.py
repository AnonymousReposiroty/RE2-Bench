from sympy.polys.domains.domain import Domain, Er, Es, Eg
from sympy.polys.monomials import monomial_min, monomial_ldiv

def dmp_terms_gcd(f: dmp[Er], u: int, K: Domain[Er]) -> tuple[tuple[int, ...], dmp[Er]]:
    """
    Remove GCD of terms from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_terms_gcd

    >>> f = ZZ.map([[1, 0], [1, 0, 0], [], []])

    >>> dmp_terms_gcd(f, 1, ZZ)
    ((2, 1), [[1], [1, 0]])

    """
    if dmp_ground_TC(f, u, K) or dmp_zero_p(f, u):
        return (0,) * (u + 1), f

    F: dict[monom, Er] = dmp_to_dict(f, u, K)
    G = monomial_min(*list(F.keys()))

    if all(g == 0 for g in G):
        return G, f

    d: dict[monom, Er] = {}

    for monom, coeff in F.items():
        d[monomial_ldiv(monom, G)] = coeff

    return G, dmp_from_dict(d, u, K)
