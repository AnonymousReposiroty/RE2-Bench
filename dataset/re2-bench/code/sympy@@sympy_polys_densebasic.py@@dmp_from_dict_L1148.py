from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_from_dict(f: dict[tuple[int, ...], Er], u: int, K: Domain[Er]) -> dmp[Er]:
    """
    Create a ``K[X]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_from_dict

    >>> dmp_from_dict({(0, 0): ZZ(3), (0, 1): ZZ(2), (2, 1): ZZ(1)}, 1, ZZ)
    [[1, 0], [], [2, 3]]
    >>> dmp_from_dict({}, 0, ZZ)
    []

    """
    if not u:
        return _dmp(dup_from_dict(f, K))
    if not f:
        return dmp_zero(u, K)

    coeffs: dict[int, dict[monom, Er]] = {}

    for monom, coeff in f.items():
        head, tail = monom[0], monom[1:]

        if head in coeffs:
            coeffs[head][tail] = coeff
        else:
            coeffs[head] = {tail: coeff}

    n = max(coeffs.keys())
    v = u - 1
    h: dmp[Er] = []

    for k in range(n, -1, -1):
        dcoeff = coeffs.get(k)

        if dcoeff is not None:
            h.append(dmp_from_dict(dcoeff, v, K))
        else:
            h.append(dmp_zero(v, K))

    return dmp_strip(h, u, K)
