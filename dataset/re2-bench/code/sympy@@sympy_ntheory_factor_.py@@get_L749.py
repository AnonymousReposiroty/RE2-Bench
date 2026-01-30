from bisect import bisect_left
from collections import defaultdict, OrderedDict
from collections.abc import MutableMapping
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from .generate import sieve, primerange, nextprime

class FactorCache(MutableMapping):
    """ Provides a cache for prime factors.
    ``factor_cache`` is pre-prepared as an instance of ``FactorCache``,
    and ``factorint`` internally references it to speed up
    the factorization of prime factors.

    While cache is automatically added during the execution of ``factorint``,
    users can also manually add prime factors independently.

    >>> from sympy import factor_cache
    >>> factor_cache[15] = 5

    Furthermore, by customizing ``get_external``,
    it is also possible to use external databases.
    The following is an example using http://factordb.com .

    .. code-block:: python

        import requests
        from sympy import factor_cache

        def get_external(self, n: int) -> list[int] | None:
            res = requests.get("http://factordb.com/api", params={"query": str(n)})
            if res.status_code != requests.codes.ok:
                return None
            j = res.json()
            if j.get("status") in ["FF", "P"]:
                return list(int(p) for p, _ in j.get("factors"))

        factor_cache.get_external = get_external

    Be aware that writing this code will trigger internet access
    to factordb.com when calling ``factorint``.

    """

    def __init__(self, maxsize: int | None=None):
        self._cache: OrderedDict[int, int] = OrderedDict()
        self.maxsize = maxsize

    @property
    def maxsize(self) -> int | None:
        """ Returns the maximum cache size; if ``None``, it is unlimited. """
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError('maxsize must be None or a non-negative integer.')
        self._maxsize = value
        if value is not None:
            while len(self._cache) > value:
                self._cache.popitem(False)

    def get(self, n: int, default=None):
        """ Return the prime factor of ``n``.
        If it does not exist in the cache, return the value of ``default``.
        """
        if n <= sieve._list[-1]:
            if sieve._list[bisect_left(sieve._list, n)] == n:
                return n
        if n in self._cache:
            self._cache.move_to_end(n)
            return self._cache[n]
        if (factors := self.get_external(n)):
            self.add(n, factors)
            return self._cache[n]
        return default

    def add(self, n: int, factors: list[int]) -> None:
        for p in sorted(factors, reverse=True):
            self[n] = p
            nz, _ = remove(n, p)
            n = int(nz)

    def get_external(self, n: int) -> list[int] | None:
        return None
