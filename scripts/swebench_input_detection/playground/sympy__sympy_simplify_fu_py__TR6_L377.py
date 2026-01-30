# Problem: sympy@@sympy_simplify_fu.py@@TR6_L377
# Module: sympy.simplify.fu
# Function: TR6
# Line: 377

from sympy.simplify.fu import TR6
from sympy import sympify

def test_input(pred_input):
    assert TR6(rv = sympify('cos(phi)'), max = 4, pow = False)==TR6(rv = sympify(pred_input['args']['rv']), max = pred_input['args']['max'], pow = pred_input['args']['pow']), 'Prediction failed!'
