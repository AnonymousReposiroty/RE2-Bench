# Problem: sympy@@sympy_matrices_reductions.py@@_row_reduce_list_L24
# Module: sympy.matrices.reductions
# Function: _row_reduce_list
# Line: 24

from sympy.matrices.reductions import _row_reduce_list


def test_input(pred_input):
    assert _row_reduce_list(mat = ['-9', '-17', '-19', '-10', '-18', '-17', '-21', '-38', '-40'], rows = 3, cols = 3, one = '1', iszerofunc = '<function _iszero at 0x755447f77240>', simpfunc = '<function _simplify at 0x755447f77380>', normalize_last = True, normalize = False, zero_above = False)==_row_reduce_list(mat = pred_input['args']['mat'], rows = pred_input['args']['rows'], cols = pred_input['args']['cols'], one = pred_input['args']['one'], iszerofunc = pred_input['args']['iszerofunc'], simpfunc = pred_input['args']['simpfunc'], normalize_last = pred_input['args']['normalize_last'], normalize = pred_input['args']['normalize'], zero_above = pred_input['args']['zero_above']), 'Prediction failed!'