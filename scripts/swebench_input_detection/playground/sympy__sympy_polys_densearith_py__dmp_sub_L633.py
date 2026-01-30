# Problem: sympy@@sympy_polys_densearith.py@@dmp_sub_L633
# Module: sympy.polys.densearith
# Function: dmp_sub
# Line: 633

from sympy.polys.densearith import dmp_sub
from sympy import ZZ
from sympy import sympify

def test_input(pred_input):
    assert dmp_sub(f = [], g = [], u = 0, K = ZZ)==dmp_sub(f = pred_input['args']['f'], g = pred_input['args']['g'], u = pred_input['args']['u'], K = sympify(pred_input['args']['K'])), 'Prediction failed!'
    
    