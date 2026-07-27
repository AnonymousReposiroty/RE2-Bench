from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_ground_LC(f: dmp[Er], u: int, K: Domain[Er]) -> Er:
    """
    Return the ground leading coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_LC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_LC(f, 2, ZZ)
    1

    """
    while u:
        f = dmp_LC(f, K)
        u -= 1

    return dup_LC(_dup(f), K)
