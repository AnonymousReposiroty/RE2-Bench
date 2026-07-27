from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_exclude(f: dmp[Er], u: int, K: Domain[Er]) -> tuple[list[int], dmp[Er], int]:
    """
    Exclude useless levels from ``f``.

    Return the levels excluded, the new excluded ``f``, and the new ``u``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_exclude

    >>> f = ZZ.map([[[1]], [[1], [2]]])

    >>> dmp_exclude(f, 2, ZZ)
    ([2], [[1], [1, 2]], 1)

    """
    if not u or dmp_ground_p(f, None, u):
        return [], f, u

    J: list[int] = []
    F = dmp_to_dict(f, u, K)

    for j in range(0, u + 1):
        for monom in F.keys():
            if monom[j]:
                break
        else:
            J.append(j)

    if not J:
        return [], f, u

    d: dict[tuple[int, ...], Er] = {}

    for monom, coeff in F.items():
        lmonom = list(monom)

        for j in reversed(J):
            del lmonom[j]

        d[tuple(lmonom)] = coeff

    u -= len(J)

    return J, dmp_from_dict(d, u, K), u
