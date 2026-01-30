from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_raise(f: dmp[Er], l: int, u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Return a multivariate polynomial raised ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_raise

    >>> f = ZZ.map([[], [1, 2]])

    >>> dmp_raise(f, 2, 1, ZZ)
    [[[[]]], [[[1]], [[2]]]]

    """
    if not l:
        return f

    if not u:
        if not f:
            return dmp_zero(l, K)

        k = l - 1

        return [dmp_ground(c, k, K) for c in _dup(f)]

    v = u - 1

    return [dmp_raise(cp, l, v, K) for cp in f]
