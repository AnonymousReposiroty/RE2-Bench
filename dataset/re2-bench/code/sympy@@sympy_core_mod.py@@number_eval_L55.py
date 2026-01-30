from .add import Add
from .exprtools import gcd_terms
from .function import DefinedFunction
from .kind import NumberKind
from .mul import Mul
from .numbers import equal_valued
from .relational import is_le, is_lt, is_ge, is_gt
from .singleton import S
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import gcd

class Mod(DefinedFunction):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """
    kind = NumberKind

    @classmethod
    def eval(cls, p, q):

        def number_eval(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """
            if q.is_zero:
                raise ZeroDivisionError('Modulo by zero')
            if p is S.NaN or q is S.NaN or p.is_finite is False or (q.is_finite is False):
                return S.NaN
            if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
                return S.Zero
            if q.is_Number:
                if p.is_Number:
                    return p % q
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One
            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv
            r = p / q
            if r.is_integer:
                return S.Zero
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if isinstance(d, int):
                    rv = p - d * q
                    if (rv * q < 0) == True:
                        rv += q
                    return rv
            if q.is_positive:
                comp1, comp2 = (is_le, is_lt)
            elif q.is_negative:
                comp1, comp2 = (is_ge, is_gt)
            else:
                return
            ls = -2 * q
            r = p - q
            for _ in range(4):
                if not comp1(ls, p):
                    return
                if comp2(r, ls):
                    return p - ls
                ls += q
        rv = number_eval(p, q)
        if rv is not None:
            return rv
        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner * (q - qinner)).is_nonnegative:
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner * (q + qinner)).is_nonpositive:
                return p
        elif isinstance(p, Add):
            both_l = non_mod_l, mod_l = ([], [])
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            if mod_l and all((inner.args[1] == q for inner in mod_l)):
                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
                return cls(net, q)
        elif isinstance(p, Mul):
            both_l = non_mod_l, mod_l = ([], [])
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            if mod_l and all((inner.args[1] == q for inner in mod_l)) and all((t.is_integer for t in p.args)) and q.is_integer:
                non_mod_l = [cls(x, q) for x in non_mod_l]
                mod = []
                non_mod = []
                for j in non_mod_l:
                    if isinstance(j, cls):
                        mod.append(j.args[0])
                    else:
                        non_mod.append(j)
                prod_mod = Mul(*mod)
                prod_non_mod = Mul(*non_mod)
                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
                net = prod_mod1 * prod_mod
                return prod_non_mod * cls(net, q)
            if q.is_Integer and q is not S.One:
                if all((t.is_integer for t in p.args)):
                    non_mod_l = [i % q if i.is_Integer else i for i in p.args]
                    if any((iq is S.Zero for iq in non_mod_l)):
                        return S.Zero
            p = Mul(*non_mod_l + mod_l)
        from sympy.polys.polyerrors import PolynomialError
        from sympy.polys.polytools import gcd
        try:
            G = gcd(p, q)
            if not equal_valued(G, 1):
                p, q = [gcd_terms(i / G, clear=False, fraction=False) for i in (p, q)]
        except PolynomialError:
            G = S.One
        pwas, qwas = (p, q)
        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)
        else:
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if equal_valued(r, 0):
                    G *= cq
                    p *= int(cp / cq)
                    ok = True
            if not ok:
                p = cp * p
                q = cq * q
        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]
        rv = number_eval(p, q)
        if rv is not None:
            return rv * G
        if G.is_Float and equal_valued(G, 1):
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and equal_valued(G.args[0], 1):
            p = G.args[0] * p
            G = Mul._from_args(G.args[1:])
        return G * cls(p, q, evaluate=(p, q) != (pwas, qwas))
