from .assumptions import StdFactKB, _assume_defined
from sympy.utilities.iterables import is_sequence, _sift_true_false
from typing import Any, Literal, Iterable, Callable

def _filter_assumptions(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    true, false = _sift_true_false(kwargs.items(), lambda i: i[0] in _assume_defined)
    assumptions = dict(true)
    nonassumptions = dict(false)
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions
