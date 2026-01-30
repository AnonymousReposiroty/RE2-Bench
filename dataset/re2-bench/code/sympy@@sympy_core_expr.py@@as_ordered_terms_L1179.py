from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
import re
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .sorting import default_sort_key
from sympy.utilities.iterables import has_variety, _sift_true_false
from .mul import Mul
from .add import Add
from typing import Any, Hashable
from typing_extensions import Self
from .numbers import Number
from sympy.polys.orderings import monomial_key
from .numbers import Number, NumberSymbol
from .exprtools import decompose_power
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.exponential import exp, log
from sympy.polys.polytools import factor
from sympy.functions.elementary.complexes import im, re

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

    @classmethod
    def _parse_order(cls, order):
        """Parse and configure the ordering of terms. """
        from sympy.polys.orderings import monomial_key
        startswith = getattr(order, 'startswith', None)
        if startswith is None:
            reverse = False
        else:
            reverse = startswith('rev-')
            if reverse:
                order = order[4:]
        monom_key = monomial_key(order)

        def neg(monom):
            return tuple([neg(m) if isinstance(m, tuple) else -m for m in monom])

        def key(term):
            _, ((re, im), monom, ncpart) = term
            monom = neg(monom_key(monom))
            ncpart = tuple([e.sort_key(order=order) for e in ncpart])
            coeff = ((bool(im), im), (re, im))
            return (monom, ncpart, coeff)
        return (key, reverse)

    def as_ordered_terms(self, order=None, data: Literal[False]=False) -> list[Expr]:
        """
        Transform an expression to an ordered list of terms.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x

        >>> (sin(x)**2*cos(x) + sin(x)**2 + 1).as_ordered_terms()
        [sin(x)**2*cos(x), sin(x)**2, 1]

        """
        from .numbers import Number, NumberSymbol
        if order is None and self.is_Add:
            key = lambda x: not isinstance(x, (Number, NumberSymbol))
            add_args = sorted(Add.make_args(self), key=key)
            if len(add_args) == 2 and isinstance(add_args[0], (Number, NumberSymbol)) and isinstance(add_args[1], Mul):
                mul_args = sorted(Mul.make_args(add_args[1]), key=key)
                if len(mul_args) == 2 and isinstance(mul_args[0], Number) and add_args[0].is_positive and mul_args[0].is_negative:
                    return add_args
        key, reverse = self._parse_order(order)
        terms, gens = self.as_terms()
        if not any((term.is_Order for term, _ in terms)):
            ordered = sorted(terms, key=key, reverse=reverse)
        else:
            _order, _terms = _sift_true_false(terms, lambda x: x[0].is_Order)
            ordered = sorted(_terms, key=key, reverse=True) + sorted(_order, key=key, reverse=True)
        if data:
            return (ordered, gens)
        else:
            return [term for term, _ in ordered]

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
