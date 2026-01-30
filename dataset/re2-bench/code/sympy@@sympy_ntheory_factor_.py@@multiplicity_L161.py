from sympy.core.numbers import Rational, Integer
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from sympy.utilities.misc import as_int, filldedent
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.factorials import factorial

def multiplicity(p, n):
    """
    Find the greatest integer m such that p**m divides n.

    Examples
    ========

    >>> from sympy import multiplicity, Rational
    >>> [multiplicity(5, n) for n in [8, 5, 25, 125, 250]]
    [0, 1, 2, 3, 3]
    >>> multiplicity(3, Rational(1, 9))
    -2

    Note: when checking for the multiplicity of a number in a
    large factorial it is most efficient to send it as an unevaluated
    factorial or to call ``multiplicity_in_factorial`` directly:

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial
    >>> p = factorial(25)
    >>> n = 2**100
    >>> nfac = factorial(n, evaluate=False)
    >>> multiplicity(p, nfac)
    52818775009509558395695966887
    >>> _ == multiplicity_in_factorial(p, n)
    True

    See Also
    ========

    trailing

    """
    try:
        p, n = as_int(p), as_int(n)
    except ValueError:
        from sympy.functions.combinatorial.factorials import factorial
        if all(isinstance(i, (SYMPY_INTS, Rational)) for i in (p, n)):
            p = Rational(p)
            n = Rational(n)
            if p.q == 1:
                if n.p == 1:
                    return -multiplicity(p.p, n.q)
                return multiplicity(p.p, n.p) - multiplicity(p.p, n.q)
            elif p.p == 1:
                return multiplicity(p.q, n.q)
            else:
                like = min(
                    multiplicity(p.p, n.p),
                    multiplicity(p.q, n.q))
                cross = min(
                    multiplicity(p.q, n.p),
                    multiplicity(p.p, n.q))
                return like - cross
        elif (isinstance(p, (SYMPY_INTS, Integer)) and
                isinstance(n, factorial) and
                isinstance(n.args[0], Integer) and
                n.args[0] >= 0):
            return multiplicity_in_factorial(p, n.args[0])
        raise ValueError(f"expecting ints or fractions, got {p} and {n}")

    if n == 0:
        raise ValueError(f"no such integer exists: multiplicity of {n} is not-defined")
    return remove(n, p)[1]
