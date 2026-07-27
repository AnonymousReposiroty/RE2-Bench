from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_gcd(f: dup[MPZ], g: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> dup[MPZ]:
    """
    Euclidean Algorithm in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_gcd

    >>> gf_gcd(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 3]

    """
    while g:
        f, g = g, gf_rem(f, g, p, K)

    return gf_monic(f, p, K)[1]
