# Problem: sympy@@sympy_matrices_matrixbase.py@@row_join_L654
# Module: sympy.matrices.matrixbase
# Function: row_join
# Line: 654

from sympy.matrices.matrixbase import MatrixBase


def test_input(pred_input):
    obj_ins = MatrixBase()
    obj_ins.rows = 2
    obj_ins.cols = 2
    obj_ins._rep = 'DomainMatrix({0: {0: 1}, 1: {1: 1}}, (2, 2), ZZ)'
    obj_ins_pred = MatrixBase()
    obj_ins_pred.rows = pred_input['self']['rows']
    obj_ins_pred.cols = pred_input['self']['cols']
    obj_ins_pred._rep = pred_input['self']['_rep']
    assert obj_ins.row_join(other = 'Matrix([\n[1, 0],\n[0, 1]])')==obj_ins_pred.row_join(other = pred_input['args']['other']), 'Prediction failed!'


