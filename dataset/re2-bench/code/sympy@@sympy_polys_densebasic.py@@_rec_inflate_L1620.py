from sympy.polys.domains.domain import Domain, Er, Es, Eg

def _rec_inflate(g: dmp[Er], M: list[int], v: int, i: int, K: Domain[Er]) -> dmp[Er]:
    """Recursive helper for :func:`dmp_inflate`."""
    if not v:
        return _dmp(dup_inflate(_dup(g), M[i], K))
    if M[i] <= 0:
        raise IndexError("all M[i] must be positive, got %s" % M[i])

    w, j = v - 1, i + 1

    g = [_rec_inflate(c, M, w, j, K) for c in g]

    result = [g[0]]

    for coeff in g[1:]:
        for _ in range(1, M[i]):
            result.append(dmp_zero(w, K))

        result.append(coeff)

    return result
