# Problem: sympy@@sympy_polys_densearith.py@@dup_abs_L443
# Module: sympy.polys.densearith
# Function: dup_abs
# Line: 443

from sympy.polys.densearith import dup_abs
from sympy import sympify

def test_input(pred_input):
    assert dup_abs(f = [2], K = sympify('ZZ'))==dup_abs(f = pred_input['args']['f'], K = sympify(pred_input['args']['K'])), 'Prediction failed!'
