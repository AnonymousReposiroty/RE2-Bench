# Problem: sympy@@sympy_polys_factortools.py@@dup_zz_factor_L639
# Module: sympy.polys.factortools
# Function: dup_zz_factor
# Line: 639

from sympy.polys.factortools import dup_zz_factor
from sympy import sympify

def test_input(pred_input):
    assert dup_zz_factor(f = [1], K = sympify('ZZ'))==dup_zz_factor(f = pred_input['args']['f'], K = sympify(pred_input['args']['K'])), 'Prediction failed!'