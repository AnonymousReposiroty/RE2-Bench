from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .logic import fuzzy_or, fuzzy_not
from .kind import NumberKind
from typing import Any, Hashable
from typing_extensions import Self
from sympy.solvers.solvers import denoms
from sympy.simplify.simplify import nsimplify, simplify

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

    def is_constant(self, *wrt, **flags):
        """Return True if self is constant, False if not, or None if
        the constancy could not be determined conclusively.

        Explanation
        ===========

        If an expression has no free symbols then it is a constant. If
        there are free symbols it is possible that the expression is a
        constant, perhaps (but not necessarily) zero. To test such
        expressions, a few strategies are tried:

        1) numerical evaluation at two random points. If two such evaluations
        give two different values and the values have a precision greater than
        1 then self is not constant. If the evaluations agree or could not be
        obtained with any precision, no decision is made. The numerical testing
        is done only if ``wrt`` is different than the free symbols.

        2) differentiation with respect to variables in 'wrt' (or all free
        symbols if omitted) to see if the expression is constant or not. This
        will not always lead to an expression that is zero even though an
        expression is constant (see added test in test_expr.py). If
        all derivatives are zero then self is constant with respect to the
        given symbols.

        3) finding out zeros of denominator expression with free_symbols.
        It will not be constant if there are zeros. It gives more negative
        answers for expression that are not constant.

        If neither evaluation nor differentiation can prove the expression is
        constant, None is returned unless two numerical values happened to be
        the same and the flag ``failing_number`` is True -- in that case the
        numerical value will be returned.

        If flag simplify=False is passed, self will not be simplified;
        the default is True since self should be simplified before testing.

        Examples
        ========

        >>> from sympy import cos, sin, Sum, S, pi
        >>> from sympy.abc import a, n, x, y
        >>> x.is_constant()
        False
        >>> S(2).is_constant()
        True
        >>> Sum(x, (x, 1, 10)).is_constant()
        True
        >>> Sum(x, (x, 1, n)).is_constant()
        False
        >>> Sum(x, (x, 1, n)).is_constant(y)
        True
        >>> Sum(x, (x, 1, n)).is_constant(n)
        False
        >>> Sum(x, (x, 1, n)).is_constant(x)
        True
        >>> eq = a*cos(x)**2 + a*sin(x)**2 - a
        >>> eq.is_constant()
        True
        >>> eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
        True

        >>> (0**x).is_constant()
        False
        >>> x.is_constant()
        False
        >>> (x**x).is_constant()
        False
        >>> one = cos(x)**2 + sin(x)**2
        >>> one.is_constant()
        True
        >>> ((one - 1)**(x + 1)).is_constant() in (True, False) # could be 0 or 1
        True
        """
        simplify = flags.get('simplify', True)
        if self.is_number:
            return True
        free = self.free_symbols
        if not free:
            return True
        wrt = set(wrt)
        if wrt and (not wrt & free):
            return True
        wrt = wrt or free
        expr = self
        if simplify:
            expr = expr.simplify()
        if expr.is_zero:
            return True
        wrt_number = {sym for sym in wrt if sym.kind is NumberKind}
        failing_number = None
        if wrt_number == free:
            try:
                a = expr.subs(list(zip(free, [0] * len(free))), simultaneous=True)
                if a is S.NaN:
                    a = expr._random(None, 0, 0, 0, 0)
            except ZeroDivisionError:
                a = None
            if a is not None and a is not S.NaN:
                try:
                    b = expr.subs(list(zip(free, [1] * len(free))), simultaneous=True)
                    if b is S.NaN:
                        b = expr._random(None, 1, 0, 1, 0)
                except ZeroDivisionError:
                    b = None
                if b is not None and b is not S.NaN and (b.equals(a) is False):
                    return False
                b = expr._random(None, -1, 0, 1, 0)
                if b is not None and b is not S.NaN and (b.equals(a) is False):
                    return False
                b = expr._random()
                if b is not None and b is not S.NaN:
                    if b.equals(a) is False:
                        return False
                    failing_number = a if a.is_number else b
        for w in wrt_number:
            deriv = expr.diff(w)
            if simplify:
                deriv = deriv.simplify()
            if deriv != 0:
                if not pure_complex(deriv, or_real=True):
                    if flags.get('failing_number', False):
                        return failing_number
                return False
        from sympy.solvers.solvers import denoms
        return fuzzy_not(fuzzy_or((den.is_zero for den in denoms(self))))
    __round__ = round

    @property
    def free_symbols(self) -> set[Basic]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol`` (see examples)

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method.

        Examples
        ========

        >>> from sympy import Derivative, Integral, IndexedBase
        >>> from sympy.abc import x, y, n
        >>> (x + 1).free_symbols
        {x}
        >>> Integral(x, y).free_symbols
        {x, y}

        Not all free symbols are actually symbols:

        >>> IndexedBase('F')[0].free_symbols
        {F, F[0]}

        The symbols of differentiation are not included unless they
        appear in the expression being differentiated.

        >>> Derivative(x + y, y).free_symbols
        {x, y}
        >>> Derivative(x, y).free_symbols
        {x}
        >>> Derivative(x, (y, n)).free_symbols
        {n, x}

        If you want to know if a symbol is in the variables of the
        Derivative you can do so as follows:

        >>> Derivative(x, y).has_free(y)
        True
        """
        empty: set[Basic] = set()
        return empty.union(*(a.free_symbols for a in self.args))

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
