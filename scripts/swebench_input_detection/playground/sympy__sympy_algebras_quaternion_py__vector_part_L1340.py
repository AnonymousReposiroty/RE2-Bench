# Problem: sympy@@sympy_algebras_quaternion.py@@vector_part_L1340
# Module: sympy.algebras.quaternion
# Function: vector_part
# Line: 1340

from sympy.algebras.quaternion import Quaternion


def test_input(pred_input):
    obj_ins = Quaternion()
    obj_ins._real_field = True
    obj_ins._norm = None
    obj_ins_pred = Quaternion()
    obj_ins_pred._real_field = pred_input['self']['_real_field']
    obj_ins_pred._norm = pred_input['self']['_norm']
    assert obj_ins.vector_part()==obj_ins_pred.vector_part(), 'Prediction failed!'
