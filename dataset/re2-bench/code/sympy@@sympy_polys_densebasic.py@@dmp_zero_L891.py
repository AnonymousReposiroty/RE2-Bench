from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_zero(u: int, K: Domain[Er] | None = None) -> dmp[Er]:
    """
    Return a multivariate zero.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero

    >>> dmp_zero(4)
    [[[[[]]]]]

    """
    r: dmp[Er] = []

    for i in range(u):
        r = [r]

    return r
