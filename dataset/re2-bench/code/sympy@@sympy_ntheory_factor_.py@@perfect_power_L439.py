import math
from sympy.core.numbers import Rational, Integer
from sympy.core.singleton import S
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from sympy.utilities.misc import as_int, filldedent

def perfect_power(n, candidates=None, big=True, factor=True):
    """
    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a
    perfect power). A ValueError is raised if ``n`` is not Rational.

    By default, the base is recursively decomposed and the exponents
    collected so the largest possible ``e`` is sought. If ``big=False``
    then the smallest possible ``e`` (thus prime) will be chosen.

    If ``factor=True`` then simultaneous factorization of ``n`` is
    attempted since finding a factor indicates the only possible root
    for ``n``. This is True by default since only a few small factors will
    be tested in the course of searching for the perfect power.

    The use of ``candidates`` is primarily for internal use; if provided,
    False will be returned if ``n`` cannot be written as a power with one
    of the candidates as an exponent and factoring (beyond testing for
    a factor of 2) will not be attempted.

    Examples
    ========

    >>> from sympy import perfect_power, Rational
    >>> perfect_power(16)
    (2, 4)
    >>> perfect_power(16, big=False)
    (4, 2)

    Negative numbers can only have odd perfect powers:

    >>> perfect_power(-4)
    False
    >>> perfect_power(-8)
    (-2, 3)

    Rationals are also recognized:

    >>> perfect_power(Rational(1, 2)**3)
    (1/2, 3)
    >>> perfect_power(Rational(-3, 2)**3)
    (-3/2, 3)

    Notes
    =====

    To know whether an integer is a perfect power of 2 use

        >>> is2pow = lambda n: bool(n and not n & (n - 1))
        >>> [(i, is2pow(i)) for i in range(5)]
        [(0, False), (1, True), (2, True), (3, False), (4, True)]

    It is not necessary to provide ``candidates``. When provided
    it will be assumed that they are ints. The first one that is
    larger than the computed maximum possible exponent will signal
    failure for the routine.

        >>> perfect_power(3**8, [9])
        False
        >>> perfect_power(3**8, [2, 4, 8])
        (3, 8)
        >>> perfect_power(3**8, [4, 8], big=False)
        (9, 4)

    See Also
    ========
    sympy.core.intfunc.integer_nthroot
    sympy.ntheory.primetest.is_square
    """
    # negative handling
    if n < 0:
        if candidates is None:
            pp = perfect_power(-n, big=True, factor=factor)
            if not pp:
                return False

            b, e = pp
            e2 = e & (-e)
            b, e = b ** e2, e // e2

            if e <= 1:
                return False

            if big or isprime(e):
                return -b, e

            for p in primerange(3, e + 1):
                if e % p == 0:
                    return - b ** (e // p), p

        odd_candidates = {i for i in candidates if i % 2}
        if not odd_candidates:
            return False

        pp = perfect_power(-n, odd_candidates, big, factor)
        if pp:
            return -pp[0], pp[1]

        return False

    # non-integer handling
    if isinstance(n, Rational) and not isinstance(n, Integer):
        p, q = n.p, n.q

        if p == 1:
            qq = perfect_power(q, candidates, big, factor)
            return (S.One / qq[0], qq[1]) if qq is not False else False

        if not (pp:=perfect_power(p, factor=factor)):
            return False
        if not (qq:=perfect_power(q, factor=factor)):
            return False
        (num_base, num_exp), (den_base, den_exp) = pp, qq

        def compute_tuple(exponent):
            """Helper to compute final result given an exponent"""
            new_num = num_base ** (num_exp // exponent)
            new_den = den_base ** (den_exp // exponent)
            return n.func(new_num, new_den), exponent

        if candidates:
            valid_candidates = [i for i in candidates
                                if num_exp % i == 0 and den_exp % i == 0]
            if not valid_candidates:
                return False

            e = max(valid_candidates) if big else min(valid_candidates)
            return compute_tuple(e)

        g = math.gcd(num_exp, den_exp)
        if g == 1:
            return False

        if big:
            return compute_tuple(g)

        e = next(p for p in primerange(2, g + 1) if g % p == 0)
        return compute_tuple(e)

    if candidates is not None:
        candidates = set(candidates)

    # positive integer handling
    n = as_int(n)

    if candidates is None and big:
        return _perfect_power(n)

    if n <= 3:
        # no unique exponent for 0, 1
        # 2 and 3 have exponents of 1
        return False
    logn = math.log2(n)
    max_possible = int(logn) + 2  # only check values less than this
    not_square = n % 10 in [2, 3, 7, 8]  # squares cannot end in 2, 3, 7, 8
    min_possible = 2 + not_square
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        candidates = sorted([i for i in candidates
            if min_possible <= i < max_possible])
        if n%2 == 0:
            e = bit_scan1(n)
            candidates = [i for i in candidates if e%i == 0]
        if big:
            candidates = reversed(candidates)
        for e in candidates:
            r, ok = iroot(n, e)
            if ok:
                return int(r), e
        return False

    def _factors():
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)

    for fac, e in zip(_factors(), candidates):
        # see if there is a factor present
        if factor and n % fac == 0:
            # find what the potential power is
            e = remove(n, fac)[1]
            # if it's a trivial power we are done
            if e == 1:
                return False

            # maybe the e-th root of n is exact
            r, exact = iroot(n, e)
            if not exact:
                # Having a factor, we know that e is the maximal
                # possible value for a root of n.
                # If n = fac**e*m can be written as a perfect
                # power then see if m can be written as r**E where
                # gcd(e, E) != 1 so n = (fac**(e//E)*r)**E
                m = n//fac**e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    r, E = rE
                    r, e = fac**(e//E)*r, E
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    r, e = r**(e//e0[0]), e0[0]
            return int(r), e

        # Weed out downright impossible candidates
        if logn/e < 40:
            b = 2.0**(logn/e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue

        # now see if the plausible e makes a perfect power
        r, exact = iroot(n, e)
        if exact:
            if big:
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    r, e = m[0], e*m[1]
            return int(r), e

    return False
