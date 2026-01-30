from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core.logic import _fuzzy_group, fuzzy_and, fuzzy_or

def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
    return _fuzzy_group(
        (ask(key(a), assumptions) for a in expr.args), quick_exit=True)
