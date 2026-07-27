# Problem: sympy@@sympy_core_function.py@@_eval_evalf_L534
# Module: sympy.core.function
# Function: _eval_evalf
# Line: 534

from sympy.core.function import Function
from sympy import symbols

def test_input(pred_input):
    x = symbols('x')  
    obj_ins = Function('f', nargs=1)
    exp = obj_ins(x)
    obj_ins_pred = Function('f', nargs=pred_input['self']['nargs'])
    exp_pred = obj_ins_pred(x)
    assert exp._eval_evalf(prec = 30)==exp_pred._eval_evalf(prec = pred_input['args']['prec']), 'Prediction failed!'