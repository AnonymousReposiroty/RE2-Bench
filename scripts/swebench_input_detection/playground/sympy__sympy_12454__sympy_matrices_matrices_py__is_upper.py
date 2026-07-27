# Problem: sympy__sympy-12454@@sympy.matrices.matrices.py@@is_upper
# Benchmark: Swebench
# Module: sympy.matrices.matrices
# Function: is_upper

from sympy.matrices.matrices import MatrixProperties


def test_input(pred_input):
    obj_ins = MatrixProperties()
    obj_ins.rows = 4
    obj_ins.cols = 4
    obj_ins._mat = None
    obj_ins_pred = MatrixProperties()
    obj_ins_pred.rows = pred_input['self']['rows']
    obj_ins_pred.cols = pred_input['self']['cols']
    obj_ins_pred._mat = pred_input['self']['_mat']
    assert obj_ins.is_upper()==obj_ins_pred.is_upper(), 'Prediction failed!'
    

