# Problem: sympy@@sympy_simplify_fu.py@@TR111_L1419
# Module: sympy.simplify.fu
# Function: TR111
# Line: 1419

from sympy.simplify.fu import TR111


def test_input(pred_input):
    assert TR111(rv = 'cos(phi)**2')==TR111(rv = pred_input['args']['rv']), 'Prediction failed!'
    