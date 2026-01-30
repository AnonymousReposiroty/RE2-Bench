# Problem: sympy@@sympy_matrices_repmatrix.py@@_getitem_RepMatrix_L1047
# Module: sympy.matrices.repmatrix
# Function: _getitem_RepMatrix
# Line: 1047

from sympy.matrices.repmatrix import _getitem_RepMatrix


def test_input(pred_input):
    assert _getitem_RepMatrix(key = [0, 0])==_getitem_RepMatrix(key = pred_input['args']['key']), 'Prediction failed!'