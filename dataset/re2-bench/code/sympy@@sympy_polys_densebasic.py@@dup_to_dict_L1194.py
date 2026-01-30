from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_to_dict(f: dup[Er], K: Domain[Er]) -> dict[tuple[int, ...], Er]:
    """
    Convert ``K[x]`` polynomial to a ``dict``.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.densebasic import dup_to_dict

    >>> dup_to_dict(ZZ.map([1, 0, 5, 0, 7]), ZZ)
    {(0,): 7, (2,): 5, (4,): 1}
    >>> dup_to_dict([], ZZ)
    {}

    .. versionchanged:: 1.15.0
        The ``zero`` parameter was removed and the ``K`` parameter is now
        required.

    """
    return {(k,): fk for k, fk in enumerate(f[::-1]) if fk}
