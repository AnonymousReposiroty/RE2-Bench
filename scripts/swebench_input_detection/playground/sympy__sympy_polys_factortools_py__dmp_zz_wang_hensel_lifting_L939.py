# Problem: sympy@@sympy_polys_factortools.py@@dmp_zz_wang_hensel_lifting_L939
# Module: sympy.polys.factortools
# Function: dmp_zz_wang_hensel_lifting
# Line: 939

from sympy.polys.factortools import dmp_zz_wang_hensel_lifting
from sympy import ZZ

def test_input(pred_input):
    assert dmp_zz_wang_hensel_lifting(f = [[1], [], [-1, 0, 0]], H = [[1, -1], [1, 1]], LC = [[1], [1]], A = [-1], p = 37, u = 1, K = ZZ)==dmp_zz_wang_hensel_lifting(f = pred_input['args']['f'], H = pred_input['args']['H'], LC = pred_input['args']['LC'], A = pred_input['args']['A'], p = pred_input['args']['p'], u = pred_input['args']['u'], K = ZZ), 'Prediction failed!'
    
