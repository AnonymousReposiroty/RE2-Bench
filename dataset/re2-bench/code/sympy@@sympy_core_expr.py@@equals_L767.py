from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .add import Add
from .power import Pow
from .mod import Mod
from .exprtools import factor_terms
from typing import Any, Hashable
from typing_extensions import Self
from sympy.simplify.simplify import nsimplify, simplify
from sympy.solvers.solvers import solve
from sympy.polys.polyerrors import NotAlgebraic
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.numberfields import minimal_polynomial
from sympy.polys.polyerrors import NotAlgebraic
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
    __round__ = round
