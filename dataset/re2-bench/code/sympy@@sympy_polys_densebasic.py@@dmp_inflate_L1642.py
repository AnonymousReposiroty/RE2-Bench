from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_inflate(f: dmp[Er], M: list[int], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Map ``y_i`` to ``x_i**k_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_inflate

    >>> f = ZZ.map([[1, 2], [3, 4]])

    >>> dmp_inflate(f, (2, 3), 1, ZZ)
    [[1, 0, 0, 2], [], [3, 0, 0, 4]]

    """
    if not u:
        return _dmp(dup_inflate(_dup(f), M[0], K))

    if all(m == 1 for m in M):
        return f
    else:
        return _rec_inflate(f, M, u, 0, K)
