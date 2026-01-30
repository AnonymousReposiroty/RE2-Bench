from sympy.polys.domains.domain import Domain, Er, Es, Eg

def _rec_strip(g: dmp[Er], v: int, K: Domain[Er] | None = None) -> dmp[Er]:
    """Recursive helper for :func:`_rec_strip`."""
    if not v:
        return _dmp(dup_strip(_dup(g), K))

    w = v - 1

    return dmp_strip([_rec_strip(c, w, K) for c in g], v, K)
