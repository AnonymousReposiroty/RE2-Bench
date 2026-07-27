from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_add(f: dup[MPZ], g: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> dup[MPZ]:
    """
    Add polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add

    >>> gf_add([3, 2, 4], [2, 2, 2], 5, ZZ)
    [4, 1]

    """
    if not f:
        return g
    if not g:
        return f

    df = gf_degree(f)
    dg = gf_degree(g)

    if df == dg:
        return gf_strip([(a + b) % p for a, b in zip(f, g)])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [(a + b) % p for a, b in zip(f, g)]
