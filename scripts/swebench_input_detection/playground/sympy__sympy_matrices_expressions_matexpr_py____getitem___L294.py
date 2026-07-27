# Problem: sympy@@sympy_matrices_expressions_matexpr.py@@__getitem___L294
# Module: sympy.matrices.expressions.matexpr
# Function: __getitem__
# Line: 294

from sympy import MatrixSymbol 

def test_input(pred_input):
    obj_ins = MatrixSymbol('X', 3, 3)
    obj_ins_pred = MatrixSymbol('X', 3, 3)
    assert obj_ins.__getitem__(key = (1, 2))==obj_ins_pred.__getitem__(key = (pred_input['args']['key'][0], pred_input['args']['key'][1])), 'Prediction failed!'
    

