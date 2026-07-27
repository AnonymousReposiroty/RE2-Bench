# Problem: sympy@@sympy_polys_polyutils.py@@_parallel_dict_from_expr_no_gens_L271
# Module: sympy.polys.polyutils
# Function: _parallel_dict_from_expr_no_gens
# Line: 271

from sympy.polys.polyutils import _parallel_dict_from_expr_no_gens


def test_input(pred_input):
    assert _parallel_dict_from_expr_no_gens(exprs = ['cos(phi)'], opt = {'fraction': True})==_parallel_dict_from_expr_no_gens(exprs = pred_input['args']['exprs'], opt = pred_input['args']['opt']), 'Prediction failed!'