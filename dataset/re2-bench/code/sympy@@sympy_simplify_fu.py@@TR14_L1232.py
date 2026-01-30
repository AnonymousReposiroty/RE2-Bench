from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def TR14(rv, first=True):
    """Convert factored powers of sin and cos identities into simpler
    expressions.

    Examples
    ========

    >>> from sympy.simplify.fu import TR14
    >>> from sympy.abc import x, y
    >>> from sympy import cos, sin
    >>> TR14((cos(x) - 1)*(cos(x) + 1))
    -sin(x)**2
    >>> TR14((sin(x) - 1)*(sin(x) + 1))
    -cos(x)**2
    >>> p1 = (cos(x) + 1)*(cos(x) - 1)
    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)
    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    >>> TR14(p1*p2*p3*(x - 1))
    -18*(x - 1)*sin(x)**2*sin(y)**4

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        if first:
            # sort them by location in numerator and denominator
            # so the code below can just deal with positive exponents
            n, d = rv.as_numer_denom()
            if d is not S.One:
                newn = TR14(n, first=False)
                newd = TR14(d, first=False)
                if newn != n or newd != d:
                    rv = newn/newd
                return rv

        other = []
        process = []
        for a in rv.args:
            if a.is_Pow:
                b, e = a.as_base_exp()
                if not (e.is_integer or b.is_positive):
                    other.append(a)
                    continue
                a = b
            else:
                e = S.One
            m = as_f_sign_1(a)
            if not m or m[1].func not in (cos, sin):
                if e is S.One:
                    other.append(a)
                else:
                    other.append(a**e)
                continue
            g, f, si = m
            process.append((g, e.is_Number, e, f, si, a))

        # sort them to get like terms next to each other
        process = list(ordered(process))

        # keep track of whether there was any change
        nother = len(other)

        # access keys
        keys = (g, t, e, f, si, a) = list(range(6))

        while process:
            A = process.pop(0)
            if process:
                B = process[0]

                if A[e].is_Number and B[e].is_Number:
                    # both exponents are numbers
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = min(A[e], B[e])

                            # reinsert any remainder
                            # the B will likely sort after A so check it first
                            if B[e] != take:
                                rem = [B[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            elif A[e] != take:
                                rem = [A[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)

                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

                elif A[e] == B[e]:
                    # both exponents are equal symbols
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = A[e]
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

            # either we are done or neither condition above applied
            other.append(A[a]**A[e])

        if len(other) != nother:
            rv = Mul(*other)

        return rv

    return bottom_up(rv, f)
