from sympy.external.gmpy import MPZ, invert

def gf_int(a: MPZ, p: MPZ) -> MPZ:
    """
    Coerce ``a mod p`` to an integer in the range ``[-p/2, p/2]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_int

    >>> gf_int(2, 7)
    2
    >>> gf_int(5, 7)
    -2

    """
    if a <= p // 2:
        return a
    else:
        return a - p
