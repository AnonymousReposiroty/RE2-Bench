from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_LC(f: dup[Er], K: Domain[Er]) -> Er:
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_LC

    >>> dup_LC([1, 2, 3], ZZ)
    1

    """
    if not f:
        return K.zero
    else:
        return f[0]
