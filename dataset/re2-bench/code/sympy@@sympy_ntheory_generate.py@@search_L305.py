from bisect import bisect, bisect_left
from array import array as _array
from sympy.external.gmpy import sqrt
from sympy.utilities.misc import as_int

class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    def __init__(self, sieve_interval=1000000):
        """ Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13])
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4])
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1])
        if sieve_interval <= 0:
            raise ValueError('sieve_interval should be a positive integer')
        self.sieve_interval = sieve_interval
        assert all((len(i) == self._n for i in (self._list, self._tlist, self._mlist)))

    def extend(self, n):
        """Grow the sieve to cover all primes <= n.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num ** 2
        while num2 <= n:
            self._list += _array('L', self._primerange(num, num2))
            num, num2 = (num2, num2 ** 2)
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """ Generate all prime numbers in the range (a, b).

        Parameters
        ==========

        a, b : positive integers assuming the following conditions
                * a is an even number
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): prime numbers such that ``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range(-(a + 1 + p) // 2 % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield (a + 2 * idx + 1)
            a += 2 * block_size

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError(f'n should be >= 2 but got: {n}')
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return (b, b)
        else:
            return (b, b + 1)
