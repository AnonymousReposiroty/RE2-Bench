from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)

def _factorint_small(factors, n, limit, fail_max, next_p=2):
    """
    Return the value of n and either a 0 (indicating that factorization up
    to the limit was complete) or else the next near-prime that would have
    been tested.

    Factoring stops if there are fail_max unsuccessful tests in a row.

    If factors of n were found they will be in the factors dictionary as
    {factor: multiplicity} and the returned value of n will have had those
    factors removed. The factors dictionary is modified in-place.

    """

    def done(n, d):
        """return n, d if the sqrt(n) was not reached yet, else
           n, 0 indicating that factoring is done.
        """
        if d*d <= n:
            return n, d
        return n, 0

    limit2 = limit**2
    threshold2 = min(n, limit2)

    if next_p < 3:
        if not n & 1:
            m = bit_scan1(n)
            factors[2] = m
            n >>= m
            threshold2 = min(n, limit2)
        next_p = 3
        if threshold2 < 9: # next_p**2 = 9
            return done(n, next_p)

    if next_p < 5:
        if not n % 3:
            n //= 3
            m = 1
            while not n % 3:
                n //= 3
                m += 1
                if m == 20:
                    n, mm = remove(n, 3)
                    m += mm
                    break
            factors[3] = m
            threshold2 = min(n, limit2)
        next_p = 5
        if threshold2 < 25: # next_p**2 = 25
            return done(n, next_p)

    # Because of the order of checks, starting from `min_p = 6k+5`,
    # useless checks are caused.
    # We want to calculate
    # next_p += [-1, -2, 3, 2, 1, 0][next_p % 6]
    p6 = next_p % 6
    next_p += (-1 if p6 < 2 else 5) - p6

    fails = 0
    while fails < fail_max:
        # next_p % 6 == 5
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 2
        if threshold2 < next_p**2:
            return done(n, next_p)

        # next_p % 6 == 1
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 4
        if threshold2 < next_p**2:
            return done(n, next_p)
    return done(n, next_p)
