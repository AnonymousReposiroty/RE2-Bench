from sympy.external.gmpy import MPZ, invert
from sympy.polys.domains.domain import Domain
from sympy.polys.densebasic import dup

def gf_quo_ground(f: dup[MPZ], a: MPZ, p: MPZ, K: Domain[MPZ]) -> dup[MPZ]:
    """
    Compute ``f/a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo_ground

    >>> gf_quo_ground(ZZ.map([3, 2, 4]), ZZ(2), 5, ZZ)
    [4, 1, 2]

    """
    return gf_mul_ground(f, K.invert(a, p), p, K)
