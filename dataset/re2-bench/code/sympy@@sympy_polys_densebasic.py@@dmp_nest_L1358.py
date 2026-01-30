from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_nest(f: dmp[Er], l: int, K: Domain[Er]) -> dmp[Er]:
    """
    Return a multivariate value nested ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nest

    >>> dmp_nest([[ZZ(1)]], 2, ZZ)
    [[[[1]]]]

    """
    if not isinstance(f, list):
        return dmp_ground(f, l, K)

    for i in range(l):
        f = [f]

    return f
