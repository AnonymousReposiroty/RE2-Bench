from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_ground_TC(f: dmp[Er], u: int, K: Domain[Er]) -> Er:
    """
    Return the ground trailing coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_TC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_TC(f, 2, ZZ)
    3

    """
    while u:
        f = dmp_TC(f, K)
        u -= 1

    return dup_TC(_dup(f), K)
