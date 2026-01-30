from sympy.utilities.iterables import (has_dups, sift, iterable,
    is_sequence, uniq, topological_sort)
import inspect

def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    eval_ = getattr(cls, 'eval', cls)

    parameters = inspect.signature(eval_).parameters.items()
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    # how many have no default and how many have a default value
    no, yes = map(len, sift(p_or_k,
        lambda p:p.default == p.empty, binary=True))
    return no if not yes else tuple(range(no, no + yes + 1))
