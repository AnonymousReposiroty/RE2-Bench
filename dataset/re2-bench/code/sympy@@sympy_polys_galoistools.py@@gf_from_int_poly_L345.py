from sympy.external.gmpy import MPZ, invert
from sympy.polys.densebasic import dup

def gf_from_int_poly(f: dup[MPZ], p: MPZ) -> dup[MPZ]:
    """
    Create a ``GF(p)[x]`` polynomial from ``Z[x]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_from_int_poly

    >>> gf_from_int_poly([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_trunc(f, p)
