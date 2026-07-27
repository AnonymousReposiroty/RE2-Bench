from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_sqf_p(f: dup[MPZ], p: MPZ, K: Domain[MPZ]) -> bool:
    """
    Return ``True`` if ``f`` is square-free in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_p

    >>> gf_sqf_p(ZZ.map([3, 2, 4]), 5, ZZ)
    True
    >>> gf_sqf_p(ZZ.map([2, 4, 4, 2, 2, 1, 4]), 5, ZZ)
    False

    """
    _, f = gf_monic(f, p, K)

    if not f:
        return True
    else:
        return gf_gcd(f, gf_diff(f, p, K), p, K) == [K.one]
