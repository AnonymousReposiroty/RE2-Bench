from sympy.external.gmpy import MPZ, invert
from sympy.polys.densebasic import dup

def gf_strip(f: dup[MPZ]) -> dup[MPZ]:
    """
    Remove leading zeros from ``f``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_strip

    >>> gf_strip([0, 0, 0, 3, 0, 1])
    [3, 0, 1]

    """
    if not f or f[0]:
        return f

    k = 0

    for coeff in f:
        if coeff:
            break
        else:
            k += 1

    return f[k:]
