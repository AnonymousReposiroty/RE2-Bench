# Problem: sympy@@sympy_polys_galoistools.py@@gf_rem_L761
# Module: sympy.polys.galoistools
# Function: gf_rem
# Line: 761

from sympy.polys.galoistools import gf_rem
from sympy import sympify

def test_input(pred_input):
    assert gf_rem(f = [2, 0], g = [1, 0, 1], p = 3, K = sympify('ZZ'))==gf_rem(f = pred_input['args']['f'], g = pred_input['args']['g'], p = pred_input['args']['p'], K = sympify(pred_input['args']['K'])), 'Prediction failed!'
    
