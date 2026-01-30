from sympy.external.gmpy import (gcd as number_gcd, lcm as number_lcm, sqrt,
                                 iroot, bit_scan1, gcdext)
from sympy.utilities.misc import as_int, filldedent

def integer_nthroot(y, n):
    """
    Return a tuple containing x = floor(y**(1/n))
    and a boolean indicating whether the result is exact (that is,
    whether x**n == y).

    Examples
    ========

    >>> from sympy import integer_nthroot
    >>> integer_nthroot(16, 2)
    (4, True)
    >>> integer_nthroot(26, 2)
    (5, False)

    To simply determine if a number is a perfect square, the is_square
    function should be used:

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(26)
    False

    See Also
    ========
    sympy.ntheory.primetest.is_square
    integer_log
    """
    x, b = iroot(as_int(y), as_int(n))
    return int(x), b
