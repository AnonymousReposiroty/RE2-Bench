from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_sqr(f: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> dup[MPZ]:
    """
    Square polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqr

    >>> gf_sqr([3, 2, 4], 5, ZZ)
    [4, 2, 3, 1, 1]

    """
    df = gf_degree(f)

    dh = 2 * df
    h = [K.zero] * (dh + 1)

    for i in range(0, dh + 1):
        coeff = K.zero

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            coeff += f[j] * f[i - j]

        coeff += coeff

        if n & 1:
            elem = f[jmax + 1]
            coeff += elem**2

        h[i] = coeff % p

    return gf_strip(h)
