# Problem: sympy@@sympy_core_mul.py@@_eval_is_zero_infinite_helper_L1306
# Module: sympy.core.mul
# Function: _eval_is_zero_infinite_helper
# Line: 1306

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("-pi/4"))
    obj_ins_pred = Mul(sympify(pred_input['self']))
    assert obj_ins._eval_is_zero_infinite_helper()==obj_ins_pred._eval_is_zero_infinite_helper(), 'Prediction failed!'
