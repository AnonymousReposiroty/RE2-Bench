from sympy.external.gmpy import MPZ, invert
from sympy.polys.densebasic import dup

def gf_trunc(f: dup[MPZ], p: MPZ) -> dup[MPZ]:
    """
    Reduce all coefficients modulo ``p``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_trunc

    >>> gf_trunc([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_strip([a % p for a in f])
