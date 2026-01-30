from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_degree_list(f: dmp[Er], u: int) -> tuple[int, ...]:
    """
    Return a list of degrees of ``f`` in ``K[X]``.

    The degree of a zero polynomial is ``-1``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_list

    >>> f = ZZ.map([[1], [1, 2, 3]])

    >>> dmp_degree_list(f, 1)
    (1, 2)

    .. versionchanged:: 1.15.0
        The degree of a zero polynomial is now ``-1`` instead of
        ``float('-inf')``.

    """
    degs = [-1] * (u + 1)
    _rec_degree_list(f, u, 0, degs)
    return tuple(degs)
