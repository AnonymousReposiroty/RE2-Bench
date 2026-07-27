from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_degree(f: dup[Er]) -> int:
    """
    Return the leading degree of ``f`` in ``K[x]``.

    Note that the degree of 0 is ``-1``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_degree

    >>> f = ZZ.map([1, 2, 0, 3])

    >>> dup_degree(f)
    3

    .. versionchanged:: 1.15.0
        The degree of a zero polynomial is now ``-1`` instead of
        ``float('-inf')``.

    """
    return len(f) - 1
