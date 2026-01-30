# Problem: sympy@@sympy_core_mul.py@@_eval_pos_neg_L1612
# Module: sympy.core.mul
# Function: _eval_pos_neg
# Line: 1612

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("-phi"))
    obj_ins_pred = Mul(sympify(pred_input['self']))
    assert obj_ins._eval_pos_neg(sign = 1)==obj_ins_pred._eval_pos_neg(sign = pred_input['args']['sign']), 'Prediction failed!'