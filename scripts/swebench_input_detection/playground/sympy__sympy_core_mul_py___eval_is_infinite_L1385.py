# Problem: sympy@@sympy_core_mul.py@@_eval_is_infinite_L1385
# Module: sympy.core.mul
# Function: _eval_is_infinite
# Line: 1385

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("pi/2"))
    obj_ins_pred = Mul(sympify(pred_input['self']))
    assert obj_ins._eval_is_infinite()==obj_ins_pred._eval_is_infinite(), 'Prediction failed!'
    
