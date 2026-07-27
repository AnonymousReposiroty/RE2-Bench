from typing import TYPE_CHECKING, overload, Any, Callable
from collections.abc import Iterable, Mapping
from sympy.core.basic import Basic
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.kind import BooleanKind, NumberKind
from sympy.assumptions import ask

@sympify_method_args
class Boolean(Basic):
    """A Boolean object is an object for which logic operations make sense."""
    __slots__ = ()
    kind = BooleanKind
    if TYPE_CHECKING:

        def __new__(cls, *args: Basic | complex) -> Boolean:
            ...

        @overload
        def subs(self, arg1: Mapping[Basic | complex, Boolean | complex], arg2: None=None) -> Boolean:
            ...

        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Boolean | complex]], arg2: None=None, **kwargs: Any) -> Boolean:
            ...

        @overload
        def subs(self, arg1: Boolean | complex, arg2: Boolean | complex) -> Boolean:
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

        def simplify(self, **kwargs) -> Boolean:
            ...
    __rand__ = __and__
    __ror__ = __or__
    __rrshift__ = __lshift__
    __rlshift__ = __rshift__
    __rxor__ = __xor__

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask
        ret = ask(self, assumptions)
        if ret is True:
            return true
        elif ret is False:
            return false
        return None
