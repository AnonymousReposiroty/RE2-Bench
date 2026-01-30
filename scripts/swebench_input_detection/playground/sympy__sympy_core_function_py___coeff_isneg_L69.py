# Problem: sympy@@sympy_core_function.py@@_coeff_isneg_L69
# Module: sympy.core.function
# Function: _coeff_isneg
# Line: 69

from sympy.core.function import _coeff_isneg
from sympy import sympify

def test_input(pred_input):
    assert _coeff_isneg(a = sympify('cos(2*phi)/2'))==_coeff_isneg(a = sympify(pred_input['args']['a'])), 'Prediction failed!'
