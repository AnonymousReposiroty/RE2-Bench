# Problem: sympy@@sympy_simplify_fu.py@@TR8_L422
# Module: sympy.simplify.fu
# Function: TR8
# Line: 422

from sympy.simplify.fu import TR8
from sympy import sympify

def test_input(pred_input):
    assert TR8(rv = sympify('cos(phi)**2'), first = False)==TR8(rv = sympify(pred_input['args']['rv']), first = pred_input['args']['first']), 'Prediction failed!'