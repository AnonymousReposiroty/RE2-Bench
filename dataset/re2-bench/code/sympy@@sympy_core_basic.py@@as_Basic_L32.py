from .sympify import _sympify, sympify, SympifyError, _external_converter
from sympy.utilities.misc import filldedent, func_name

def as_Basic(expr):
    """Return expr as a Basic instance using strict sympify
    or raise a TypeError; this is just a wrapper to _sympify,
    raising a TypeError instead of a SympifyError."""
    try:
        return _sympify(expr)
    except SympifyError:
        raise TypeError(
            'Argument must be a Basic object, not `%s`' % func_name(
            expr))
