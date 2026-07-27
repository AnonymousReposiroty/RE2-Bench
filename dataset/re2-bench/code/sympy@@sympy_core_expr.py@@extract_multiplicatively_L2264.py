from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .mul import Mul
from .add import Add
from .power import Pow
from typing import Any, Hashable
from typing_extensions import Self
from .numbers import Number
from sympy.functions.elementary.complexes import conjugate as c
from .numbers import Number, NumberSymbol
from .add import _unevaluated_Add
from sympy.functions.elementary.exponential import exp
from .add import _unevaluated_Add
from sympy.functions.elementary.exponential import exp, log

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

    def as_base_exp(self) -> tuple[Expr, Expr]:
        return (self, S.One)

    def primitive(self) -> tuple[Number, Expr]:
        """Return the positive Rational that can be extracted non-recursively
        from every term of self (i.e., self is treated like an Add). This is
        like the as_coeff_Mul() method but primitive always extracts a positive
        Rational (never a negative or a Float).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (3*(x + 1)**2).primitive()
        (3, (x + 1)**2)
        >>> a = (6*x + 2); a.primitive()
        (2, 3*x + 1)
        >>> b = (x/2 + 3); b.primitive()
        (1/2, x + 6)
        >>> (a*b).primitive() == (1, a*b)
        True
        """
        if not self:
            return (S.One, S.Zero)
        c, r = self.as_coeff_Mul(rational=True)
        if c.is_negative:
            c, r = (-c, -r)
        return (c, r)

    def extract_multiplicatively(self, c: Expr | complex) -> Expr | None:
        """Return None if it's not possible to make self in the form
           c * something in a nice way, i.e. preserving the properties
           of arguments of self.

           Examples
           ========

           >>> from sympy import symbols, Rational

           >>> x, y = symbols('x,y', real=True)

           >>> ((x*y)**3).extract_multiplicatively(x**2 * y)
           x*y**2

           >>> ((x*y)**3).extract_multiplicatively(x**4 * y)

           >>> (2*x).extract_multiplicatively(2)
           x

           >>> (2*x).extract_multiplicatively(3)

           >>> (Rational(1, 2)*x).extract_multiplicatively(3)
           x/6

        """
        from sympy.functions.elementary.exponential import exp
        from .add import _unevaluated_Add
        c = sympify(c)
        if self is S.NaN:
            return None
        if c is S.One:
            return self
        elif c == self:
            return S.One
        if c.is_Add:
            cc, pc = c.primitive()
            if cc is not S.One:
                c = Mul(cc, pc, evaluate=False)
        if c.is_Mul:
            a, b = c.as_two_terms()
            x = self.extract_multiplicatively(a)
            if x is not None:
                return x.extract_multiplicatively(b)
            else:
                return x
        quotient = self / c
        if self.is_Number:
            if self is S.Infinity:
                if c.is_positive:
                    return S.Infinity
            elif self is S.NegativeInfinity:
                if c.is_negative:
                    return S.Infinity
                elif c.is_positive:
                    return S.NegativeInfinity
            elif self is S.ComplexInfinity:
                if not c.is_zero:
                    return S.ComplexInfinity
            elif self.is_Integer:
                if not quotient.is_Integer:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Rational:
                if not quotient.is_Rational:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Float:
                if not quotient.is_Float:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
        elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
            if quotient.is_Mul and len(quotient.args) == 2:
                if quotient.args[0].is_Integer and quotient.args[0].is_positive and (quotient.args[1] == self):
                    return quotient
            elif quotient.is_Integer and c.is_Number:
                return quotient
        elif self.is_Add:
            cs, ps = self.primitive()
            if c.is_Number and c is not S.NegativeOne:
                if cs is not S.One:
                    if c.is_negative:
                        xc = cs.extract_multiplicatively(-c)
                        if xc is not None:
                            xc = -xc
                    else:
                        xc = cs.extract_multiplicatively(c)
                    if xc is not None:
                        return xc * ps
                return None
            if c == ps:
                return cs
            newargs = []
            arg: Expr
            for arg in ps.args:
                newarg = arg.extract_multiplicatively(c)
                if newarg is None:
                    return None
                newargs.append(newarg)
            if cs is not S.One:
                args = [cs * t for t in newargs]
                return _unevaluated_Add(*args)
            else:
                return Add._from_args(newargs)
        elif self.is_Mul:
            args: list[Expr] = list(self.args)
            for i, arg in enumerate(args):
                newarg = arg.extract_multiplicatively(c)
                if newarg is not None:
                    args[i] = newarg
                    return Mul(*args)
        elif self.is_Pow or isinstance(self, exp):
            sb, se = self.as_base_exp()
            cb, ce = c.as_base_exp()
            if cb == sb:
                new_exp = se.extract_additively(ce)
                if new_exp is not None:
                    return Pow(sb, new_exp)
            elif c == sb:
                new_exp = se.extract_additively(1)
                if new_exp is not None:
                    return Pow(sb, new_exp)
        return None

    @overload
    def as_coeff_Mul(self, rational: Literal[True]) -> tuple['Rational', Expr]:
        ...

    @overload
    def as_coeff_Mul(self, rational: bool=False) -> tuple['Number', Expr]:
        ...

    def as_coeff_Mul(self, rational: bool=False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a product."""
        return (S.One, self)
    __round__ = round

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
