from sympy.external.gmpy import MPZ, invert
from sympy.polys.densebasic import dup

def gf_degree(f: dup[MPZ]) -> int:
    """
    Return the leading degree of ``f``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_degree

    >>> gf_degree([1, 1, 2, 0])
    3
    >>> gf_degree([])
    -1

    """
    return len(f) - 1
