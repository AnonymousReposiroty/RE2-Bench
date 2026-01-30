from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.intfunc import num_digits
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.factorials import factorial

def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              use_ecm=True, verbose=False, visual=None, multiple=False):
    r"""
    Given a positive integer ``n``, ``factorint(n)`` returns a dict containing
    the prime factors of ``n`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy.ntheory import factorint
    >>> factorint(2000)    # 2000 = (2**4) * (5**3)
    {2: 4, 5: 3}
    >>> factorint(65537)   # This number is prime
    {65537: 1}

    For input less than 2, factorint behaves as follows:

        - ``factorint(1)`` returns the empty factorization, ``{}``
        - ``factorint(0)`` returns ``{0:1}``
        - ``factorint(-n)`` adds ``-1:1`` to the factors and then factors ``n``

    Partial Factorization:

    If ``limit`` (> 3) is specified, the search is stopped after performing
    trial division up to (and including) the limit (or taking a
    corresponding number of rho/p-1 steps). This is useful if one has
    a large number and only is interested in finding small factors (if
    any). Note that setting a limit does not prevent larger factors
    from being found early; it simply means that the largest factor may
    be composite. Since checking for perfect power is relatively cheap, it is
    done regardless of the limit setting.

    This number, for example, has two small factors and a huge
    semi-prime factor that cannot be reduced easily:

    >>> from sympy.ntheory import isprime
    >>> a = 1407633717262338957430697921446883
    >>> f = factorint(a, limit=10000)
    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}
    True
    >>> isprime(max(f))
    False

    This number has a small factor and a residual perfect power whose
    base is greater than the limit:

    >>> factorint(3*101**7, limit=5)
    {3: 1, 101: 7}

    List of Factors:

    If ``multiple`` is set to ``True`` then a list containing the
    prime factors including multiplicities is returned.

    >>> factorint(24, multiple=True)
    [2, 2, 2, 3]

    Visual Factorization:

    If ``visual`` is set to ``True``, then it will return a visual
    factorization of the integer.  For example:

    >>> from sympy import pprint
    >>> pprint(factorint(4200, visual=True))
     3  1  2  1
    2 *3 *5 *7

    Note that this is achieved by using the evaluate=False flag in Mul
    and Pow. If you do other manipulations with an expression where
    evaluate=False, it may evaluate.  Therefore, you should use the
    visual option only for visualization, and use the normal dictionary
    returned by visual=False if you want to perform operations on the
    factors.

    You can easily switch between the two forms by sending them back to
    factorint:

    >>> from sympy import Mul
    >>> regular = factorint(1764); regular
    {2: 2, 3: 2, 7: 2}
    >>> pprint(factorint(regular))
     2  2  2
    2 *3 *7

    >>> visual = factorint(1764, visual=True); pprint(visual)
     2  2  2
    2 *3 *7
    >>> print(factorint(visual))
    {2: 2, 3: 2, 7: 2}

    If you want to send a number to be factored in a partially factored form
    you can do so with a dictionary or unevaluated expression:

    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form
    {2: 10, 3: 3}
    >>> factorint(Mul(4, 12, evaluate=False))
    {2: 4, 3: 1}

    The table of the output logic is:

        ====== ====== ======= =======
                       Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    mul    dict    mul
        n       mul    dict    dict
        mul     mul    dict    dict
        ====== ====== ======= =======

    Notes
    =====

    Algorithm:

    The function switches between multiple algorithms. Trial division
    quickly finds small factors (of the order 1-5 digits), and finds
    all large factors if given enough time. The Pollard rho and p-1
    algorithms are used to find large factors ahead of time; they
    will often find factors of the order of 10 digits within a few
    seconds:

    >>> factors = factorint(12345678910111213141516)
    >>> for base, exp in sorted(factors.items()):
    ...     print('%s %s' % (base, exp))
    ...
    2 2
    2507191691 1
    1231026625769 1

    Any of these methods can optionally be disabled with the following
    boolean parameters:

        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method

    ``factorint`` also periodically checks if the remaining part is
    a prime number or a perfect power, and in those cases stops.

    For unevaluated factorial, it uses Legendre's formula(theorem).


    If ``verbose`` is set to ``True``, detailed progress is printed.

    See Also
    ========

    smoothness, smoothness_p, divisors

    """
    if isinstance(n, Dict):
        n = dict(n)
    if multiple:
        fac = factorint(n, limit=limit, use_trial=use_trial,
                           use_rho=use_rho, use_pm1=use_pm1,
                           verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p in sorted(fac)), [])
        return factorlist

    factordict = {}
    if visual and not isinstance(n, (Mul, dict)):
        factordict = factorint(n, limit=limit, use_trial=use_trial,
                               use_rho=use_rho, use_pm1=use_pm1,
                               verbose=verbose, visual=False)
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for k, v in
            n.as_powers_dict().items()}
    elif isinstance(n, dict):
        factordict = n
    if factordict and isinstance(n, (Mul, dict)):
        # check it
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho,
                          use_pm1=use_pm1, verbose=verbose, visual=False)
            for k, v in d.items():
                if k in factordict:
                    factordict[k] += v*e
                else:
                    factordict[k] = v*e
    if visual or (type(n) is dict and
                  visual is not True and
                  visual is not False):
        if factordict == {}:
            return S.One
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    elif isinstance(n, (dict, Mul)):
        return factordict

    assert use_trial or use_rho or use_pm1 or use_ecm

    from sympy.functions.combinatorial.factorials import factorial
    if isinstance(n, factorial):
        x = as_int(n.args[0])
        if x >= 20:
            factors = {}
            m = 2 # to initialize the if condition below
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    m, q = 0, x // p
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            if verbose:
                print(complete_msg)
            return factors
        else:
            # if n < 20!, direct computation is faster
            # since it uses a lookup table
            n = n.func(x)

    n = as_int(n)
    if limit:
        limit = int(limit)
        use_ecm = False

    # special cases
    if n < 0:
        factors = factorint(
            -n, limit=limit, use_trial=use_trial, use_rho=use_rho,
            use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors

    if limit and limit < 2:
        if n == 1:
            return {}
        return {n: 1}
    elif n < 10:
        # doing this we are assured of getting a limit > 2
        # when we have to compute it later
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1},
                {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]

    factors = {}

    # do simplistic factorization
    if verbose:
        sn = str(n)
        if len(sn) > 50:
            print(f"Factoring {sn[:5]}..({len(sn) - 10} other digits)..{sn[-5:]}")
        else:
            print('Factoring', n)

    # this is the preliminary factorization for small factors
    # We want to guarantee that there are no small prime factors,
    # so we run even if `use_trial` is False.
    small = 2**15
    fail_max = 600
    small = min(small, limit or small)
    if verbose:
        print(trial_int_msg % (2, small, fail_max))
    n, next_p = _factorint_small(factors, n, small, fail_max)
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    if next_p == 0:
        if n > 1:
            factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return factors
    # Check if it exists in the cache
    while p := factor_cache.get(n):
        n, e = remove(n, p)
        factors[int(p)] = int(e)
    # first check if the simplistic run didn't finish
    # because of the limit and check for a perfect
    # power before exiting
    if limit and next_p > limit:
        if verbose:
            print('Exceeded limit:', limit)
        if _check_termination(factors, n, limit, use_trial,
                              use_rho, use_pm1, verbose, next_p):
            return factors
        if n > 1:
            factors[int(n)] = 1
        return factors
    if _check_termination(factors, n, limit, use_trial,
                          use_rho, use_pm1, verbose, next_p):
        return factors

    # continue with more advanced factorization methods
    # ...do a Fermat test since it's so easy and we need the
    # square root anyway. Finding 2 factors is easy if they are
    # "close enough." This is the big root equivalent of dividing by
    # 2, 3, 5.
    sqrt_n = isqrt(n)
    a = sqrt_n + 1
    # If `n % 4 == 1`, `a` must be odd for `a**2 - n` to be a square number.
    if (n % 4 == 1) ^ (a & 1):
        a += 1
    a2 = a**2
    b2 = a2 - n
    for _ in range(3):
        b, fermat = sqrtrem(b2)
        if not fermat:
            if verbose:
                print(fermat_msg)
            for r in [a - b, a + b]:
                facs = factorint(r, limit=limit, use_trial=use_trial,
                                 use_rho=use_rho, use_pm1=use_pm1,
                                 verbose=verbose)
                for k, v in facs.items():
                    factors[k] = factors.get(k, 0) + v
            if verbose:
                print(complete_msg)
            return factors
        b2 += (a + 1) << 2  # equiv to (a + 2)**2 - n
        a += 2

    # these are the limits for trial division which will
    # be attempted in parallel with pollard methods
    low, high = next_p, 2*next_p

    # add 1 to make sure limit is reached in primerange calls
    _limit = (limit or sqrt_n) + 1
    iteration = 0
    while 1:
        high_ = min(high, _limit)

        # Trial division
        if use_trial:
            if verbose:
                print(trial_msg % (low, high_))
            ps = sieve.primerange(low, high_)
            n, found_trial = _trial(factors, n, ps, verbose)
            next_p = high_
            if found_trial and _check_termination(factors, n, limit, use_trial,
                                                  use_rho, use_pm1, verbose, next_p):
                return factors
        else:
            found_trial = False

        if high > _limit:
            if verbose:
                print('Exceeded limit:', _limit)
            if n > 1:
                factors[int(n)] = 1
            if verbose:
                print(complete_msg)
            return factors

        # Only used advanced methods when no small factors were found
        if not found_trial:
            # Pollard p-1
            if use_pm1:
                if verbose:
                    print(pm1_msg % (low, high_))
                c = pollard_pm1(n, B=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors

            # Pollard rho
            if use_rho:
                if verbose:
                    print(rho_msg % (1, low, high_))
                c = pollard_rho(n, retries=1, max_steps=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors
        # Use subexponential algorithms if use_ecm
        # Use pollard algorithms for finding small factors for 3 iterations
        # if after small factors the number of digits of n >= 25 then use ecm
        iteration += 1
        if use_ecm and iteration >= 3 and num_digits(n) >= 24:
            break
        low, high = high, high*2

    B1 = 10000
    B2 = 100*B1
    num_curves = 50
    while(1):
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        factor = _ecm_one_factor(n, B1, B2, num_curves, seed=B1)
        if factor:
            if factor < next_p**2 or isprime(factor):
                ps = [factor]
            else:
                ps = factorint(factor, limit=limit,
                           use_trial=use_trial,
                           use_rho=use_rho,
                           use_pm1=use_pm1,
                           use_ecm=use_ecm,
                           verbose=verbose)
            n, _ = _trial(factors, n, ps, verbose=False)
            if _check_termination(factors, n, limit, use_trial,
                                  use_rho, use_pm1, verbose, next_p):
                return factors
        B1 *= 5
        B2 = 100*B1
        num_curves *= 4
