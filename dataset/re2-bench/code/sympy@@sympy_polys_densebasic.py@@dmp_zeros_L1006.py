from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_zeros(n: int, u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Return a list of multivariate zeros.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_zeros

    >>> dmp_zeros(3, 2, ZZ)
    [[[[]]], [[[]]], [[[]]]]
    >>> dmp_zeros(3, -1, ZZ)
    [0, 0, 0]

    """
    if not n:
        return []

    if u < 0:
        return _dmp([K.zero] * n)
    else:
        return [dmp_zero(u, K) for i in range(n)]
