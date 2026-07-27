from typing import TYPE_CHECKING, TypeVar, Callable, Any, cast
from sympy.polys.domains.domain import Domain, Er, Es, Eg

def dup_convert(f: dup[Er], K0: Domain[Er] | None, K1: Domain[Es]) -> dup[Es]:
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_convert

    >>> R, x = ring("x", ZZ)

    >>> dup_convert([R(1), R(2)], R.to_domain(), ZZ)
    [1, 2]
    >>> dup_convert([ZZ(1), ZZ(2)], ZZ, R.to_domain())
    [1, 2]

    """
    if K0 is not None and K0 == K1:
        return cast('dup[Es]', f)
    else:
        return dup_strip([K1.convert(c, K0) for c in f], K1)
