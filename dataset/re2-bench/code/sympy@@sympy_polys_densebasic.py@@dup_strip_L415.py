from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_strip(f: dup[Er], K: Domain[Er] | None = None) -> dup[Er]:
    """
    Remove leading zeros from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.densebasic import dup_strip

    >>> dup_strip([0, 0, 1, 2, 3, 0], ZZ)
    [1, 2, 3, 0]

    """
    if not f or f[0]:
        return f

    i = 0

    for cf in f:
        if cf:
            break
        else:
            i += 1

    return f[i:]
