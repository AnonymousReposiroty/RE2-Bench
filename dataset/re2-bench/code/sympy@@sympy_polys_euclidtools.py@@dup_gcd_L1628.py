def dup_gcd(f, g, K):
    """
    Computes polynomial GCD of `f` and `g` in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gcd(x**2 - 1, x**2 - 3*x + 2)
    x - 1

    """
    return dup_inner_gcd(f, g, K)[0]
