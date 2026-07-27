# Problem: sympy@@sympy_core_mul.py@@_eval_is_zero_L1373
# Module: sympy.core.mul
# Function: _eval_is_zero
# Line: 1373

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("pi/2"))
    obj_ins_pred = Mul(sympify(pred_input['self']["__repr__"]))
    assert obj_ins._eval_is_zero()==obj_ins_pred._eval_is_zero(), 'Prediction failed!'