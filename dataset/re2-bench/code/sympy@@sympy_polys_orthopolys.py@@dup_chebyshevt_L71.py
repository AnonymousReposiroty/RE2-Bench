def dup_chebyshevt(n, K):
    """Low-level implementation of Chebyshev polynomials of the first kind."""
    if n < 1:
        return [K.one]
    # When n is small, it is faster to directly calculate the recurrence relation.
    if n < 64: # The threshold serves as a heuristic
        return _dup_chebyshevt_rec(n, K)
    return _dup_chebyshevt_prod(n, K)
