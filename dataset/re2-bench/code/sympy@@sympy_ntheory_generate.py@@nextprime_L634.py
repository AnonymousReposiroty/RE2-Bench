from .primetest import isprime
from sympy.utilities.misc import as_int

def nextprime(n, ith=1):
    """ Return the ith prime greater than n.

        Parameters
        ==========

        n : integer
        ith : positive integer

        Returns
        =======

        int : Return the ith prime greater than n

        Raises
        ======

        ValueError
            If ``ith <= 0``.
            If ``n`` or ``ith`` is not an integer.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import nextprime
        >>> [(i, nextprime(i)) for i in range(10, 15)]
        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]
        >>> nextprime(2, ith=2) # the 2nd prime after 2
        5

        See Also
        ========

        prevprime : Return the largest prime smaller than n
        primerange : Generate all primes in a given range

    """
    n = int(n)
    i = as_int(ith)
    if i <= 0:
        raise ValueError("ith should be positive")
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:
        l, _ = sieve.search(n)
        if l + i - 1 < len(sieve._list):
            return sieve._list[l + i - 1]
        n = sieve._list[-1]
        i += l - len(sieve._list)
    nn = 6*(n//6)
    if nn == n:
        n += 1
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
