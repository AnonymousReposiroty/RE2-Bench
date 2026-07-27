from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_frobenius_monomial_base(g: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> list[dup[MPZ]]:
    """
    return the list of ``x**(i*p) mod g in Z_p`` for ``i = 0, .., n - 1``
    where ``n = gf_degree(g)``

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base
    >>> g = ZZ.map([1, 0, 2, 1])
    >>> gf_frobenius_monomial_base(g, 5, ZZ)
    [[1], [4, 4, 2], [1, 2]]

    """
    n = gf_degree(g)
    if n == 0:
        return []
    b = [[K.zero]] * n
    b[0] = [K.one]
    if p < n:
        for i in range(1, n):
            mon = gf_lshift(b[i - 1], p, K)
            b[i] = gf_rem(mon, g, p, K)
    elif n > 1:
        b[1] = gf_pow_mod([K.one, K.zero], p, g, p, K)
        for i in range(2, n):
            b[i] = gf_mul(b[i - 1], b[1], p, K)
            b[i] = gf_rem(b[i], g, p, K)

    return b
