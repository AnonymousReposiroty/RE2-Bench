from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.core.relational import Relational
from sympy.core.relational import Relational, _canonical
from sympy.core.relational import Relational
from sympy.core.relational import Relational
from sympy.core.relational import Equality, Relational
from sympy.core.relational import Relational

class Equivalent(BooleanFunction):
    """
    Equivalence relation.

    ``Equivalent(A, B)`` is True iff A and B are both True or both False.

    Returns True if all of the arguments are logically equivalent.
    Returns False otherwise.

    For two arguments, this is equivalent to :py:class:`~.Xnor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Equivalent, And
    >>> from sympy.abc import x
    >>> Equivalent(False, False, False)
    True
    >>> Equivalent(True, False, False)
    False
    >>> Equivalent(x, And(x, True))
    True

    """

    def __new__(cls, *args, evaluate=None, **kwargs):
        if evaluate is None:
            evaluate = global_parameters.evaluate
        if not evaluate:
            return super().__new__(cls, *args, evaluate=evaluate, **kwargs)
        from sympy.core.relational import Relational
        args = [_sympify(arg) for arg in args]
        argset = set(args)
        for x in args:
            if isinstance(x, Number) or x in [True, False]:
                argset.discard(x)
                argset.add(bool(x))
        rel = []
        for r in argset:
            if isinstance(r, Relational):
                rel.append((r, r.canonical, r.negated.canonical))
        remove = []
        for i, (r, c, nc) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                if cj == nc:
                    return false
                elif cj == c:
                    remove.append((r, rj))
                    break
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
            argset.add(True)
        if len(argset) <= 1:
            return true
        if True in argset:
            argset.discard(True)
            return And(*argset)
        if False in argset:
            argset.discard(False)
            return And(*[Not(arg) for arg in argset])
        return super().__new__(cls, *ordered(argset))
