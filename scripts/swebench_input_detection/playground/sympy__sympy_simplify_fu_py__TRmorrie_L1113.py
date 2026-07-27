# Problem: sympy@@sympy_simplify_fu.py@@TRmorrie_L1113
# Module: sympy.simplify.fu
# Function: TRmorrie
# Line: 1113

from sympy.simplify.fu import TRmorrie
from sympy import sympify

def test_input(pred_input):
    assert TRmorrie(rv = sympify('cos(phi)'))==TRmorrie(rv = sympify(pred_input['args']['rv'])), 'Prediction failed!'