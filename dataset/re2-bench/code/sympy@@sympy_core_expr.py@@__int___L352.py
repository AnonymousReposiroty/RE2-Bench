from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from sympy.utilities.misc import as_int, func_name, filldedent
from mpmath.libmp import mpf_log, prec_to_dps
from .add import Add
from .power import Pow
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal, int_valued
from typing import Any, Hashable
from typing_extensions import Self
from sympy.simplify.simplify import nsimplify, simplify
from sympy.solvers.solvers import solve
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
from sympy.functions.elementary.complexes import im, sign
from sympy.simplify.simplify import nsimplify

@sympify_method_args
class Expr(Basic, EvalfMixin):
    """
    Base class for algebraic expressions.

    Explanation
    ===========

    Everything that requires arithmetic operations to be defined
    should subclass this class, instead of Basic (which should be
    used only for argument storage and expression manipulation, i.e.
    pattern matching, substitutions, etc).

    If you want to override the comparisons of expressions:
    Should use _eval_is_ge for inequality, or _eval_is_eq, with multiple dispatch.
    _eval_is_ge return true if x >= y, false if x < y, and None if the two types
    are not comparable or the comparison is indeterminate

    See Also
    ========

    sympy.core.basic.Basic
    """
    __slots__: tuple[str, ...] = ()
    if TYPE_CHECKING:

        def __new__(cls, *args: Basic) -> Self:
            ...

        @overload
        def subs(self, arg1: Mapping[Basic | complex, Expr | complex], arg2: None=None) -> Expr:
            ...

        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Expr | complex]], arg2: None=None, **kwargs: Any) -> Expr:
            ...

        @overload
        def subs(self, arg1: Expr | complex, arg2: Expr | complex) -> Expr:
            ...

        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic:
            ...

        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic:
            ...

        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic:
            ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, arg2: Basic | complex | None=None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Expr:
            ...

        def evalf(self, n: int | None=15, subs: dict[Basic, Basic | float] | None=None, maxn: int=100, chop: bool | int=False, strict: bool=False, quad: str | None=None, verbose: bool=False) -> Expr:
            ...
        n = evalf
    is_scalar = True
    _op_priority = 10.0

    def __int__(self) -> int:
        if not self.is_number:
            raise TypeError('Cannot convert symbols to int')
        if not self.is_comparable:
            raise TypeError('Cannot convert non-comparable expression to int')
        r = self.round(2)
        if not r.is_Number:
            raise TypeError('Cannot convert complex to int')
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError('Cannot convert %s to int' % r)
        i = int(r)
        if not i:
            return i
        if int_valued(r):
            r = self.round(15)
            i = int(r)
        if int_valued(r):
            if (self > i) is S.true:
                return i
            if (self < i) is S.true:
                return i - 1
            ok = self.equals(i)
            if ok is None:
                raise TypeError('cannot compute int value accurately')
            if ok:
                return i
            return i - (1 if i > 0 else -1)
        return i

    @property
    def is_number(self):
        """Returns True if ``self`` has no free symbols and no
        undefined functions (AppliedUndef, to be precise). It will be
        faster than ``if not self.free_symbols``, however, since
        ``is_number`` will fail as soon as it hits a free symbol
        or undefined function.

        Examples
        ========

        >>> from sympy import Function, Integral, cos, sin, pi
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> x.is_number
        False
        >>> f(1).is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        Not all numbers are Numbers in the SymPy sense:

        >>> pi.is_number, pi.is_Number
        (True, False)

        If something is a number it should evaluate to a number with
        real and imaginary parts that are Numbers; the result may not
        be comparable, however, since the real and/or imaginary part
        of the result may not have precision.

        >>> cos(1).is_number and cos(1).is_comparable
        True

        >>> z = cos(1)**2 + sin(1)**2 - 1
        >>> z.is_number
        True
        >>> z.is_comparable
        False

        See Also
        ========

        sympy.core.basic.Basic.is_comparable
        """
        return all((obj.is_number for obj in self.args))

    def equals(self, other, failing_expression=False):
        """Return True if self == other, False if it does not, or None. If
        failing_expression is True then the expression which did not simplify
        to a 0 will be returned instead of None.

        Explanation
        ===========

        If ``self`` is a Number (or complex number) that is not zero, then
        the result is False.

        If ``self`` is a number and has not evaluated to zero, evalf will be
        used to test whether the expression evaluates to zero. If it does so
        and the result has significance (i.e. the precision is either -1, for
        a Rational result, or is greater than 1) then the evalf value will be
        used to return True or False.

        """
        from sympy.simplify.simplify import nsimplify, simplify
        from sympy.solvers.solvers import solve
        from sympy.polys.polyerrors import NotAlgebraic
        from sympy.polys.numberfields import minimal_polynomial
        other = sympify(other)
        if not isinstance(other, Expr):
            return False
        if self == other:
            return True
        diff = factor_terms(simplify(self - other), radical=True)
        if not diff:
            return True
        if not diff.has(Add, Mod):
            return False
        factors = diff.as_coeff_mul()[1]
        if len(factors) > 1:
            fac_zero = [fac.equals(0) for fac in factors]
            if None not in fac_zero:
                return any(fac_zero)
        constant = diff.is_constant(simplify=False, failing_number=True)
        if constant is False:
            return False
        if not diff.is_number:
            if constant is None:
                return
        if constant is True:
            ndiff = diff._random()
            if ndiff and ndiff.is_comparable:
                return False
        if diff.is_number:
            surds = [s for s in diff.atoms(Pow) if s.args[0].is_Integer]
            surds.sort(key=lambda x: -x.args[0])
            for s in surds:
                try:
                    sol = solve(diff, s, simplify=False)
                    if sol:
                        if s in sol:
                            return True
                        if all((si.is_Integer for si in sol)):
                            return False
                        if all((i.is_algebraic is False for i in sol)):
                            return False
                        if any((si in surds for si in sol)):
                            return False
                        if any((nsimplify(s - si) == 0 and simplify(s - si) == 0 for si in sol)):
                            return True
                        if s.is_real:
                            if any((nsimplify(si, [s]) == s and simplify(si) == s for si in sol)):
                                return True
                except NotImplementedError:
                    pass
            if True:
                try:
                    mp = minimal_polynomial(diff)
                    if mp.is_Symbol:
                        return True
                    return False
                except (NotAlgebraic, NotImplementedError):
                    pass
        if constant not in (True, None) and constant != 0:
            return False
        if failing_expression:
            return diff
        return None

    def round(self, n=None):
        """Return x rounded to the given decimal place.

        If a complex number would result, apply round to the real
        and imaginary components of the number.

        Examples
        ========

        >>> from sympy import pi, E, I, S, Number
        >>> pi.round()
        3
        >>> pi.round(2)
        3.14
        >>> (2*pi + E*I).round()
        6 + 3*I

        The round method has a chopping effect:

        >>> (2*pi + I/10).round()
        6
        >>> (pi/10 + 2*I).round()
        2*I
        >>> (pi/10 + E*I).round(2)
        0.31 + 2.72*I

        Notes
        =====

        The Python ``round`` function uses the SymPy ``round`` method so it
        will always return a SymPy number (not a Python float or int):

        >>> isinstance(round(S(123), -2), Number)
        True
        """
        x = self
        if not x.is_number:
            raise TypeError('Cannot round symbolic expression')
        if not x.is_Atom:
            if not pure_complex(x.n(2), or_real=True):
                raise TypeError('Expected a number but got %s:' % func_name(x))
        elif x in _illegal:
            return x
        if not (xr := x.is_extended_real):
            r, i = x.as_real_imag()
            if xr is False:
                return r.round(n) + S.ImaginaryUnit * i.round(n)
            if i.equals(0):
                return r.round(n)
        if not x:
            return S.Zero if n is None else x
        p = as_int(n or 0)
        if x.is_Integer:
            return Integer(round(int(x), p))
        digits_to_decimal = _mag(x)
        allow = digits_to_decimal + p
        precs = [f._prec for f in x.atoms(Float)]
        dps = prec_to_dps(max(precs)) if precs else None
        if dps is None:
            dps = max(15, allow)
        else:
            allow = min(allow, dps)
        shift = -digits_to_decimal + dps
        extra = 1
        xf = x.n(dps + extra) * Pow(10, shift)
        if xf.is_Number and xf._prec == 1:
            if x.equals(0):
                return Float(0)
            raise ValueError('not computing with precision')
        xi = Integer(xf)
        sign = 1 if x > 0 else -1
        dif2 = sign * (xf - xi).n(extra)
        if dif2 < 0:
            raise NotImplementedError('not expecting int(x) to round away from 0')
        if dif2 > 0.5:
            xi += sign
        elif dif2 == 0.5:
            xi += sign if xi % 2 else -sign
        ip = p - shift
        xr = round(xi.p, ip)
        rv = Rational(xr, Pow(10, shift))
        if rv.is_Integer:
            if n is None:
                return rv
            return Float(str(rv), dps)
        else:
            if not allow and rv > self:
                allow += 1
            return Float(rv, allow)
    __round__ = round

    @property
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        return self._eval_is_comparable()

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args
