# Problem: sympy@@sympy_core_mul.py@@_eval_expand_mul_L944
# Module: sympy.core.mul
# Function: _eval_expand_mul
# Line: 944

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("-sin(phi)**2"))
    obj_ins_pred = Mul(sympify(pred_input['self']["__repr__"]))
    assert obj_ins._eval_expand_mul(power_base = True, power_exp = True, mul = True, log = True, multinomial = True, basic = True)==obj_ins_pred._eval_expand_mul(power_base = pred_input['kwargs']['power_base'], power_exp = pred_input['kwargs']['power_exp'], mul = pred_input['kwargs']['mul'], log = pred_input['kwargs']['log'], multinomial = pred_input['kwargs']['multinomial'], basic = pred_input['kwargs']['basic']), 'Prediction failed!'
    
