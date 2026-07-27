from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .sorting import default_sort_key
from .mul import Mul
from .add import Add
from typing import Any, Hashable
from typing_extensions import Self
from .exprtools import decompose_power
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.exponential import exp, log
from sympy.polys.polytools import factor

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

    def as_terms(self) -> tuple[list[tuple[Expr, Any]], list[Expr]]:
        """Transform an expression to a list of terms. """
        from .exprtools import decompose_power
        gens_set, terms = (set(), [])
        for term in Add.make_args(self):
            coeff, _term = term.as_coeff_Mul()
            coeff_complex = complex(coeff)
            cpart, ncpart = ({}, [])
            if _term is not S.One:
                for factor in Mul.make_args(_term):
                    if factor.is_number:
                        try:
                            coeff_complex *= complex(factor)
                        except (TypeError, ValueError):
                            pass
                        else:
                            continue
                    if factor.is_commutative:
                        base, exp = decompose_power(factor)
                        cpart[base] = exp
                        gens_set.add(base)
                    else:
                        ncpart.append(factor)
            coeff_tuple = (coeff_complex.real, coeff_complex.imag)
            ncpart_tuple = tuple(ncpart)
            terms.append((term, (coeff_tuple, cpart, ncpart_tuple)))
        gens = sorted(gens_set, key=default_sort_key)
        k, indices = (len(gens), {})
        for i, g in enumerate(gens):
            indices[g] = i
        result = []
        for term, (coeff_tuple, cpart, ncpart_tuple) in terms:
            monom = [0] * k
            for base, exp in cpart.items():
                monom[indices[base]] = exp
            result.append((term, (coeff_tuple, tuple(monom), ncpart_tuple)))
        return (result, gens)
    __round__ = round
