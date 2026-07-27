# Problem: sympy@@sympy_core_mul.py@@as_real_imag_L870
# Module: sympy.core.mul
# Function: as_real_imag
# Line: 870

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("2*pi"))
    obj_ins_pred = Mul(sympify(pred_input['self']["__repr__"]))
    assert obj_ins.as_real_imag(deep = True)==obj_ins_pred.as_real_imag(deep = pred_input['args']['deep']), 'Prediction failed!'
