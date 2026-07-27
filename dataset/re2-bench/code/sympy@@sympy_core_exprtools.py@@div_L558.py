from .mul import Mul, _keep_coeff
from .basic import Basic
from .expr import Expr
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polytools import gcd, factor

class Factors:
    """Efficient representation of ``f_1*f_2*...*f_n``."""
    __slots__ = ('factors', 'gens')

    def __init__(self, factors=None):
        """Initialize Factors from dict or expr.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x
        >>> from sympy import I
        >>> e = 2*x**3
        >>> Factors(e)
        Factors({2: 1, x: 3})
        >>> Factors(e.as_powers_dict())
        Factors({2: 1, x: 3})
        >>> f = _
        >>> f.factors  # underlying dictionary
        {2: 1, x: 3}
        >>> f.gens  # base of each factor
        frozenset({2, x})
        >>> Factors(0)
        Factors({0: 1})
        >>> Factors(I)
        Factors({I: 1})

        Notes
        =====

        Although a dictionary can be passed, only minimal checking is
        performed: powers of -1 and I are made canonical.

        """
        if isinstance(factors, (SYMPY_INTS, float)):
            factors = S(factors)
        if isinstance(factors, Factors):
            factors = factors.factors.copy()
        elif factors in (None, S.One):
            factors = {}
        elif factors is S.Zero or factors == 0:
            factors = {S.Zero: S.One}
        elif isinstance(factors, Number):
            n = factors
            factors = {}
            if n < 0:
                factors[S.NegativeOne] = S.One
                n = -n
            if n is not S.One:
                if n.is_Float or n.is_Integer or n is S.Infinity:
                    factors[n] = S.One
                elif n.is_Rational:
                    if n.p != 1:
                        factors[Integer(n.p)] = S.One
                    factors[Integer(n.q)] = S.NegativeOne
                else:
                    raise ValueError('Expected Float|Rational|Integer, not %s' % n)
        elif isinstance(factors, Basic) and (not factors.args):
            factors = {factors: S.One}
        elif isinstance(factors, Expr):
            c, nc = factors.args_cnc()
            i = c.count(I)
            for _ in range(i):
                c.remove(I)
            factors = dict(Mul._from_args(c).as_powers_dict())
            for f in list(factors.keys()):
                if isinstance(f, Rational) and (not isinstance(f, Integer)):
                    p, q = (Integer(f.p), Integer(f.q))
                    factors[p] = (factors[p] if p in factors else S.Zero) + factors[f]
                    factors[q] = (factors[q] if q in factors else S.Zero) - factors[f]
                    factors.pop(f)
            if i:
                factors[I] = factors.get(I, S.Zero) + i
            if nc:
                factors[Mul(*nc, evaluate=False)] = S.One
        else:
            factors = factors.copy()
            handle = [k for k in factors if k is I or k in (-1, 1)]
            if handle:
                i1 = S.One
                for k in handle:
                    if not _isnumber(factors[k]):
                        continue
                    i1 *= k ** factors.pop(k)
                if i1 is not S.One:
                    for a in i1.args if i1.is_Mul else [i1]:
                        if a is S.NegativeOne:
                            factors[a] = S.One
                        elif a is I:
                            factors[I] = S.One
                        elif a.is_Pow:
                            factors[a.base] = factors.get(a.base, S.Zero) + a.exp
                        elif equal_valued(a, 1):
                            factors[a] = S.One
                        elif equal_valued(a, -1):
                            factors[-a] = S.One
                            factors[S.NegativeOne] = S.One
                        else:
                            raise ValueError('unexpected factor in i1: %s' % a)
        self.factors = factors
        keys = getattr(factors, 'keys', None)
        if keys is None:
            raise TypeError('expecting Expr or dictionary')
        self.gens = frozenset(keys())

    @property
    def is_zero(self):
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(0).is_zero
        True
        """
        f = self.factors
        return len(f) == 1 and S.Zero in f

    def div(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        This is optimized for the case when there are many factors in common.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> from sympy import S

        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a.div(a)
        (Factors({}), Factors({}))
        >>> a.div(x*z)
        (Factors({y: 2}), Factors({z: 1}))

        The ``/`` operator only gives ``quo``:

        >>> a/x
        Factors({y: 2})

        Factors treats its factors as though they are all in the numerator, so
        if you violate this assumption the results will be correct but will
        not strictly correspond to the numerator and denominator of the ratio:

        >>> a.div(x/z)
        (Factors({y: 2}), Factors({z: -1}))

        Factors is also naive about bases: it does not attempt any denesting
        of Rational-base terms, for example the following does not become
        2**(2*x)/2.

        >>> Factors(2**(2*x + 2)).div(S(8))
        (Factors({2: 2*x + 2}), Factors({8: 1}))

        factor_terms can clean up such Rational-bases powers:

        >>> from sympy import factor_terms
        >>> n, d = Factors(2**(2*x + 2)).div(S(8))
        >>> n.as_expr()/d.as_expr()
        2**(2*x + 2)/8
        >>> factor_terms(_)
        2**(2*x)/2

        """
        quo, rem = (dict(self.factors), {})
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                raise ZeroDivisionError
            if self.is_zero:
                return (Factors(S.Zero), Factors())
        for factor, exp in other.factors.items():
            if factor in quo:
                d = quo[factor] - exp
                if _isnumber(d):
                    if d <= 0:
                        del quo[factor]
                    if d >= 0:
                        if d:
                            quo[factor] = d
                        continue
                    exp = -d
                else:
                    r = quo[factor].extract_additively(exp)
                    if r is not None:
                        if r:
                            quo[factor] = r
                        else:
                            del quo[factor]
                    else:
                        other_exp = exp
                        sc, sa = quo[factor].as_coeff_Add()
                        if sc:
                            oc, oa = other_exp.as_coeff_Add()
                            diff = sc - oc
                            if diff > 0:
                                quo[factor] -= oc
                                other_exp = oa
                            elif diff < 0:
                                quo[factor] -= sc
                                other_exp = oa - diff
                            else:
                                quo[factor] = sa
                                other_exp = oa
                        if other_exp:
                            rem[factor] = other_exp
                        else:
                            assert factor not in rem
                    continue
            rem[factor] = exp
        return (Factors(quo), Factors(rem))
