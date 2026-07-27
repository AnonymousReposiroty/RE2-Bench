from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_LC(f: dmp[Er], K: Domain[Er]) -> dmp[Er]:
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_LC

    >>> f = ZZ.map([[1], [2, 3]])

    >>> dmp_LC(f, ZZ)
    [1]

    """
    if not f:
        return _ground_dmp(K.zero)
    else:
        return f[0]
