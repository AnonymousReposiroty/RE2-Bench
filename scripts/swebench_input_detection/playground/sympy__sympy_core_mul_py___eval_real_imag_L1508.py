# Problem: sympy@@sympy_core_mul.py@@_eval_real_imag_L1508
# Module: sympy.core.mul
# Function: _eval_real_imag
# Line: 1508

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("11*pi/6"))
    obj_ins_pred = Mul(sympify(pred_input['self']["__repr__"]))
    assert obj_ins._eval_real_imag(real = True)==obj_ins_pred._eval_real_imag(real = pred_input['args']['real']), 'Prediction failed!'
