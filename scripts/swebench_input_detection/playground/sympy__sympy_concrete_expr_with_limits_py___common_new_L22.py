# Problem: sympy@@sympy_concrete_expr_with_limits.py@@_common_new_L22
# Module: sympy.concrete.expr.with.limits
# Function: _common_new
# Line: 22

from sympy.concrete.expr_with_limits import _common_new
from sympy import sympify, symbols
from sympy import symbols, Integral

def test_input(pred_input):
    assert _common_new(Integral, 1, symbols('x'), discrete = False)==_common_new(cls = Integral, function = pred_input['args']['function'], discrete = pred_input['args']['discrete'], symbols = symbols(pred_input['args']['symbols'])), 'Prediction failed!'
    


