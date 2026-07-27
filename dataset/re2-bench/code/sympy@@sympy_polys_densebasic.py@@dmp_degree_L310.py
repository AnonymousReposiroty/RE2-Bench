from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_degree(f: dmp[Er], u: int) -> int:
    """
    Return the leading degree of ``f`` in ``x_0`` in ``K[X]``.

    Note that the degree of 0 is ``-1``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree

    >>> dmp_degree([[[]]], 2)
    -1

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree(f, 1)
    1

    .. versionchanged:: 1.15.0
        The degree of a zero polynomial is now ``-1`` instead of
        ``float('-inf')``.

    """
    if dmp_zero_p(f, u):
        return -1
    else:
        return len(f) - 1
