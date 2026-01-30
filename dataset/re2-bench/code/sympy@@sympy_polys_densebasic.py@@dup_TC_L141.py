from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_TC(f: dup[Er], K: Domain[Er]) -> Er:
    """
    Return the trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_TC

    >>> dup_TC([1, 2, 3], ZZ)
    3

    """
    if not f:
        return K.zero
    else:
        return f[-1]
