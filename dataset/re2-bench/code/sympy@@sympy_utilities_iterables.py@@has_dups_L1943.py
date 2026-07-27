from sympy.core.containers import Dict
from sympy.sets.sets import Set

def has_dups(seq):
    """Return True if there are any duplicate elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_dups, Dict, Set
    >>> has_dups((1, 2, 1))
    True
    >>> has_dups(range(3))
    False
    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))
    True
    """
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()
    try:
        return any(True for s in seq if s in unique or unique.add(s))
    except TypeError:
        return len(seq) != len(list(uniq(seq)))
