from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_from_dict(f: dict[tuple[int, ...], Er], K: Domain[Er]) -> dup[Er]:
    """
    Create a ``K[x]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_dict

    >>> dup_from_dict({(0,): ZZ(7), (2,): ZZ(5), (4,): ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]
    >>> dup_from_dict({}, ZZ)
    []

    """
    if not f:
        return []

    (n,) = max(f.keys())
    h = [K.zero] * (n + 1)

    for (k,), fk in f.items():
        h[n - k] = fk

    return dup_strip(h, K)
