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

    def normal(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        The only differences between this and method ``div`` is that this
        is 1) optimized for the case when there are few factors in common and
        2) this does not raise an error if ``other`` is zero.

        See Also
        ========
        div

        """
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                return (Factors(), Factors(S.Zero))
            if self.is_zero:
                return (Factors(S.Zero), Factors())
        self_factors = dict(self.factors)
        other_factors = dict(other.factors)
        for factor, self_exp in self.factors.items():
            try:
                other_exp = other.factors[factor]
            except KeyError:
                continue
            exp = self_exp - other_exp
            if not exp:
                del self_factors[factor]
                del other_factors[factor]
            elif _isnumber(exp):
                if exp > 0:
                    self_factors[factor] = exp
                    del other_factors[factor]
                else:
                    del self_factors[factor]
                    other_factors[factor] = -exp
            else:
                r = self_exp.extract_additively(other_exp)
                if r is not None:
                    if r:
                        self_factors[factor] = r
                        del other_factors[factor]
                    else:
                        del self_factors[factor]
                        del other_factors[factor]
                else:
                    sc, sa = self_exp.as_coeff_Add()
                    if sc:
                        oc, oa = other_exp.as_coeff_Add()
                        diff = sc - oc
                        if diff > 0:
                            self_factors[factor] -= oc
                            other_exp = oa
                        elif diff < 0:
                            self_factors[factor] -= sc
                            other_factors[factor] -= sc
                            other_exp = oa - diff
                        else:
                            self_factors[factor] = sa
                            other_exp = oa
                    if other_exp:
                        other_factors[factor] = other_exp
                    else:
                        del other_factors[factor]
        return (Factors(self_factors), Factors(other_factors))
