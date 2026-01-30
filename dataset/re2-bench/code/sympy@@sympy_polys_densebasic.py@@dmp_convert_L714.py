from typing import TYPE_CHECKING, TypeVar, Callable, Any, cast
from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dmp_convert(f: dmp[Er], u: int, K0: Domain[Er] | None, K1: Domain[Es]) -> dmp[Es]:
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_convert

    >>> R, x = ring("x", ZZ)

    >>> dmp_convert([[R(1)], [R(2)]], 1, R.to_domain(), ZZ)
    [[1], [2]]
    >>> dmp_convert([[ZZ(1)], [ZZ(2)]], 1, ZZ, R.to_domain())
    [[1], [2]]

    """
    if not u:
        return _dmp(dup_convert(_dup(f), K0, K1))
    if K0 is not None and K0 == K1:
        return cast('dmp[Es]', f)

    v = u - 1

    return dmp_strip([dmp_convert(c, v, K0, K1) for c in f], u, K1)
