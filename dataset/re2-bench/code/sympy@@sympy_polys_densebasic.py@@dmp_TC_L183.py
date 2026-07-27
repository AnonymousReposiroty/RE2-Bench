from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_TC(f: dmp[Er], K: Domain[Er]) -> dmp[Er]:
    """
    Return the trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_TC

    >>> f = ZZ.map([[1], [2, 3]])

    >>> dmp_TC(f, ZZ)
    [2, 3]

    """
    if not f:
        return _ground_dmp(K.zero)
    else:
        return f[-1]
