from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_quo(f: dup[MPZ], g: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> dup[MPZ]:
    """
    Compute exact quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo

    >>> gf_quo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1, 1]
    >>> gf_quo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return []

    inv = K.invert(g[0], p)

    h, dq, dr = f[:], df - dg, dg - 1

    for i in range(0, dq + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        h[i] = (coeff * inv) % p

    return h[: dq + 1]
