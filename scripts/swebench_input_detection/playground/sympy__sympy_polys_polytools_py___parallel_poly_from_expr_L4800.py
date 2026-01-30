# Problem: sympy@@sympy_polys_polytools.py@@_parallel_poly_from_expr_L4800
# Module: sympy.polys.polytools
# Function: _parallel_poly_from_expr
# Line: 4800

from sympy.polys.polytools import _parallel_poly_from_expr
from sympy.polys import polyoptions

def test_input(pred_input):
    opt = polyoptions.build_options([], {}) 
    assert _parallel_poly_from_expr(exprs = ['2', '-1*2*cos(phi)'], opt = opt)==_parallel_poly_from_expr(exprs = pred_input['args']['exprs'], opt = polyoptions.build_options([], pred_input['args']['opt'])), 'Prediction failed!'

