# Problem: sympy@@sympy_matrices_determinant.py@@_find_reasonable_pivot_L30
# Module: sympy.matrices.determinant
# Function: _find_reasonable_pivot
# Line: 30

from sympy.matrices.determinant import _find_reasonable_pivot
from sympy.core.expr import Expr
def _simplify(expr):
    """ Wrapper to avoid circular imports. """
    from sympy.simplify.simplify import simplify
    return simplify(expr)

def _iszero(x: Expr) -> bool | None:
    """Returns True if x is zero."""
    return getattr(x, 'is_zero', None)

def test_input(pred_input):
    assert _find_reasonable_pivot(col = ['-9', '-10', '-21'], iszerofunc = _iszero, simpfunc = _simplify)==_find_reasonable_pivot(col = pred_input['args']['col'], iszerofunc = _iszero, simpfunc = _simplify), 'Prediction failed!'
