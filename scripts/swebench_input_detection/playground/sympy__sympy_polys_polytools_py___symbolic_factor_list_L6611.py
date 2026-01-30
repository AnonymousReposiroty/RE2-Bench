# Problem: sympy@@sympy_polys_polytools.py@@_symbolic_factor_list_L6611
# Module: sympy.polys.polytools
# Function: _symbolic_factor_list
# Line: 6611

from sympy.polys.polytools import _symbolic_factor_list
from sympy import sympify
from sympy.polys.polyoptions import Options

def test_input(pred_input):
    assert sympify('cos(phi)') == sympify(pred_input['args']['expr']), 'Prediction failed!'
    
