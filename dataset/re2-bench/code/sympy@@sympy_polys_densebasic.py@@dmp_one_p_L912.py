from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_one_p(f: dmp[Er], u: int, K: Domain[Er]) -> bool:
    """
    Return ``True`` if ``f`` is one in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one_p

    >>> dmp_one_p([[[ZZ(1)]]], 2, ZZ)
    True

    """
    return dmp_ground_p(f, K.one, u)
