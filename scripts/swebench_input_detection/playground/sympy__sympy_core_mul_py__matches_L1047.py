# Problem: sympy@@sympy_core_mul.py@@matches_L1047
# Module: sympy.core.mul
# Function: matches
# Line: 1047

from sympy.core.mul import Mul
from sympy import sympify

def test_input(pred_input):
    obj_ins = Mul(sympify("sin(x*a_)**n_*cos(x*a_)**m_"))
    obj_ins_pred = Mul(sympify(pred_input['self']["__repr__"]))
    assert obj_ins.matches(expr = sympify('cos(x)'), repl_dict = None, old = False)==obj_ins_pred.matches(expr = sympify(pred_input['args']['expr']), repl_dict = pred_input['args']['repl_dict'], old = pred_input['args']['old']), 'Prediction failed!'
