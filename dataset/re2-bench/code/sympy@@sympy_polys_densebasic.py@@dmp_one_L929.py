from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_one(u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Return a multivariate one over ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one

    >>> dmp_one(2, ZZ)
    [[[1]]]

    """
    return dmp_ground(K.one, u, K)
