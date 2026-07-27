# Problem: sympy@@sympy_core_exprtools.py@@_mask_nc_L1272
# Module: sympy.core.exprtools
# Function: _mask_nc
# Line: 1272

from sympy.core.exprtools import _mask_nc
from sympy import sympify

def test_input(pred_input):
    assert _mask_nc(eq = sympify('2 - 2*cos(phi)'), name = None)==_mask_nc(eq = sympify(pred_input['args']['eq']), name = pred_input['args']['name']), 'Prediction failed!'