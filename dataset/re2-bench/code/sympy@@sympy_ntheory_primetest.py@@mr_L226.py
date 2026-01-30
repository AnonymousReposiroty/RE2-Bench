from sympy.external.gmpy import (gmpy as _gmpy, gcd, jacobi,
                                 is_square as gmpy_is_square,
                                 bit_scan1, is_fermat_prp, is_euler_prp,
                                 is_selfridge_prp, is_strong_selfridge_prp,
                                 is_strong_bpsw_prp)
from sympy.utilities.misc import as_int, filldedent
from sympy.polys.domains import ZZ
from sympy.ntheory.generate import sieve as s

def mr(n, bases):
    """Perform a Miller-Rabin strong pseudoprime test on n using a
    given list of bases/witnesses.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 135-138

    A list of thresholds and the bases they require are here:
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants

    Examples
    ========

    >>> from sympy.ntheory.primetest import mr
    >>> mr(1373651, [2, 3])
    False
    >>> mr(479001599, [31, 73])
    True

    """
    from sympy.polys.domains import ZZ

    n = as_int(n)
    if n < 2 or (n > 2 and n % 2 == 0):
        return False
    # remove powers of 2 from n-1 (= t * 2**s)
    s = bit_scan1(n - 1)
    t = n >> s
    for base in bases:
        # Bases >= n are wrapped, bases < 2 are invalid
        if base >= n:
            base %= n
        if base >= 2:
            base = ZZ(base)
            if not _test(n, base, s, t):
                return False
    return True
