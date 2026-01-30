# Problem: sympy@@sympy_polys_densearith.py@@dup_l1_norm_L1945
# Module: sympy.polys.densearith
# Function: dup_l1_norm
# Line: 1945

from sympy.polys.densearith import dup_l1_norm
from sympy import sympify

def test_input(pred_input):
    assert dup_l1_norm(f = [1, 2],  K = sympify('ZZ'))==dup_l1_norm(f = pred_input['args']['f'], K = sympify(pred_input['args']['K'])), 'Prediction failed!'
    