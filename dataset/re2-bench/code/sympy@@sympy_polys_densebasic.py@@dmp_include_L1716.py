from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_include(f: dmp[Er], J: list[int], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Include useless levels in ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_include

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_include(f, [2], 1, ZZ)
    [[[1]], [[1], [2]]]

    """
    if not J:
        return f

    F: dict[tuple[int, ...], Er] = dmp_to_dict(f, u, K)
    d: dict[tuple[int, ...], Er] = {}

    for monom, coeff in F.items():
        lmonom = list(monom)

        for j in J:
            lmonom.insert(j, 0)

        d[tuple(lmonom)] = coeff

    u += len(J)

    return dmp_from_dict(d, u, K)
