from typing import TYPE_CHECKING, overload, Any, Callable
from sympy.core.operations import AssocOp, LatticeOp

class Or(LatticeOp, BooleanFunction):
    """
    Logical OR function

    It evaluates its arguments in order, returning true immediately
    when an  argument is true, and false if they are all false.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Or
    >>> x | y
    x | y

    Notes
    =====

    The ``|`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    or. Hence, ``Or(a, b)`` and ``a | b`` will return different things if
    ``a`` and ``b`` are integers.

    >>> Or(x, y).subs(x, 0)
    y

    """
    zero = true
    identity = false
    if TYPE_CHECKING:

        def __new__(cls, *args: Boolean | bool, evaluate: bool | None=None) -> Boolean:
            ...

        @property
        def args(self) -> tuple[Boolean, ...]:
            ...

    def _eval_subs(self, old, new):
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                if bad is None:
                    bad = i
                continue
            if i == True:
                return true
            elif i != False:
                args.append(i)
        if bad is not None:
            bad.subs(old, new)
        if isinstance(old, Or):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)
        return self.func(*args)
