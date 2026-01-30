# Problem: sympy@@sympy_core_exprtools.py@@factor_terms_L1158
# Module: sympy.core.exprtools
# Function: factor_terms
# Line: 1158

from sympy.core.exprtools import factor_terms
from sympy import sympify


def test_input(pred_input):
    assert factor_terms(expr = sympify('sin(phi)'), radical = False, clear = False, fraction = False, sign = True)==factor_terms(expr = sympify(pred_input['args']['expr']), radical = pred_input['args']['radical'], clear = pred_input['args']['clear'], fraction = pred_input['args']['fraction'], sign = pred_input['args']['sign']), 'Prediction failed!'
    

