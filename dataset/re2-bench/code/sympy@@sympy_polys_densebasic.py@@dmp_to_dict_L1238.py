from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_to_dict(f: dmp[Er], u: int, K: Domain[Er]) -> dict[tuple[int, ...], Er]:
    """
    Convert a ``K[X]`` polynomial to a ``dict````.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.densebasic import dmp_to_dict

    >>> dmp_to_dict([[1, 0], [], [2, 3]], 1, ZZ)
    {(0, 0): 3, (0, 1): 2, (2, 1): 1}
    >>> dmp_to_dict([], 0, ZZ)
    {}

    .. versionchanged:: 1.15.0
        The ``zero`` parameter was removed and the ``K`` parameter is now
        required.

    """
    if not u:
        return dup_to_dict(_dup(f), K)

    n = dmp_degree(f, u)
    v = u - 1
    result: dict[monom, Er] = {}

    for k in range(0, n + 1):
        h = dmp_to_dict(f[n - k], v, K)

        for exp, coeff in h.items():
            result[(k,) + exp] = coeff

    return result
