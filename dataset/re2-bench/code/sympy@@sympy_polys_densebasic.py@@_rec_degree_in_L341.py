from sympy.polys.domains.domain import Domain, Er, Es, Eg

def _rec_degree_in(g: dmp[Er], v: int, i: int, j: int) -> int:
    """Recursive helper function for :func:`dmp_degree_in`."""
    if i == j:
        return dmp_degree(g, v)

    v, i = v - 1, i + 1

    return max(_rec_degree_in(c, v, i, j) for c in g)
