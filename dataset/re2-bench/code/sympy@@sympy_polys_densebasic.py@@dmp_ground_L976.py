from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_ground(c: Er, u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Return a multivariate constant.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.densebasic import dmp_ground

    >>> dmp_ground(3, 5, ZZ)
    [[[[[[3]]]]]]
    >>> dmp_ground(1, -1, ZZ)
    1

    """
    if not c:
        return dmp_zero(u, K)

    if u < 0:
        return _ground_dmp(c)

    f = _dmp([c])

    for i in range(u):
        f = [f]

    return f
