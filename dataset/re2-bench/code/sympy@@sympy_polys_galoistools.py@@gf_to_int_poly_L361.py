from sympy.external.gmpy import MPZ, invert
from sympy.polys.densebasic import dup

def gf_to_int_poly(f: dup[MPZ], p: MPZ, symmetric: bool = True) -> dup[MPZ]:
    """
    Convert a ``GF(p)[x]`` polynomial to ``Z[x]``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_int_poly

    >>> gf_to_int_poly([2, 3, 3], 5)
    [2, -2, -2]
    >>> gf_to_int_poly([2, 3, 3], 5, symmetric=False)
    [2, 3, 3]

    """
    if symmetric:
        return [gf_int(c, p) for c in f]
    else:
        return f
