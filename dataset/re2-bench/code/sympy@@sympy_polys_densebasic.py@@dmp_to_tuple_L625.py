from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_to_tuple(f: dmp[Er], u: int) -> dmp_tup[Er]:
    """
    Convert `f` into a nested tuple of tuples.

    This is needed for hashing.  This is similar to dmp_copy().

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_to_tuple

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_to_tuple(f, 1)
    ((1,), (1, 2))

    """
    if not u:
        return _dmp_tup(tuple(_dup(f)))
    v = u - 1

    return tuple(dmp_to_tuple(c, v) for c in f)
