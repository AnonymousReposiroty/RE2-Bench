from sympy.polys.densearith import (dup_mul, dup_mul_ground,
    dup_lshift, dup_sub, dup_add, dup_sub_term, dup_sub_ground, dup_sqr)

def _dup_chebyshevt_rec(n, K):
    r""" Chebyshev polynomials of the first kind using recurrence.

    Explanation
    ===========

    Chebyshev polynomials of the first kind are defined by the recurrence
    relation:

    .. math::
        T_0(x) &= 1\\
        T_1(x) &= x\\
        T_n(x) &= 2xT_{n-1}(x) - T_{n-2}(x)

    This function calculates the Chebyshev polynomial of the first kind using
    the above recurrence relation.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    m2, m1 = [K.one], [K.one, K.zero]
    for _ in range(n - 1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    return m1
