from sympy.polys.domains.domain import Domain, Er, Eg, Ef, Eeuclid
from sympy.polys.densebasic import (
    dup, dmp, _dup, _dmp, _dmp2, _ground_dmp,
    dup_strip, dmp_strip, dup_truncate,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_to_dict,
    dmp_from_dict,
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC, dmp_TC,
    dmp_zero, dmp_ground,
    dmp_zero_p,
    dup_to_raw_dict, dup_from_raw_dict,
    dmp_to_raw_dict,
    dmp_zeros,
    dmp_include,
    dup_nth,
)

def dup_trunc(f: dup[Eeuclid], p: Eeuclid, K: Domain[Eeuclid]) -> dup[Eeuclid]:
    """
    Reduce a ``K[x]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_trunc(2*x**3 + 3*x**2 + 5*x + 7, ZZ(3))
    -x**3 - x + 1

    """
    if K.is_ZZ:
        g: list[Eeuclid] = []

        for c in f:
            c = c % p

            if c > p // 2: # type: ignore
                g.append(c - p)
            else:
                g.append(c)
    elif K.is_FiniteField:
        # XXX: python-flint's nmod does not support %
        pi = int(p) # type: ignore
        g = [ K(int(c) % pi) for c in f ] # type: ignore
    else:
        g = [ c % p for c in f ]

    return dup_strip(g, K)
