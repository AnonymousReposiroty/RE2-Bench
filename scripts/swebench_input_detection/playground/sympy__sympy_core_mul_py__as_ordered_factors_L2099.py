# Problem: sympy@@sympy_core_mul.py@@as_ordered_factors_L2099
# Module: sympy.core.mul
# Function: as_ordered_factors
# Line: 2099

from sympy.core.mul import  Mul
from sympy import sympify
from sympy.abc import w, z  

def test_input(pred_input):
    obj_ins = Mul(sympify("w*z"))
    obj_ins_pred = Mul(sympify(pred_input['self']))
    assert obj_ins.as_ordered_factors(order = None)==obj_ins_pred.as_ordered_factors(order = pred_input['args']['order']), 'Prediction failed!'
    