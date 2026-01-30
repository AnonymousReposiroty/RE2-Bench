# Problem: sympy@@sympy_core_power.py@@_eval_is_rational_L1244
# Module: sympy.core.power
# Function: _eval_is_rational
# Line: 1244

from sympy.core.power import Pow
from sympy import sympify

def test_input(pred_input):
    obj_ins = Pow(sympify("cos(phi)"), sympify("2"))
    obj_ins_pred = Pow(sympify(pred_input['self']["__repr__"].split("**")[0]), sympify(pred_input['self']["__repr__"].split("**")[1]))
    assert obj_ins._eval_is_rational()==obj_ins_pred._eval_is_rational(), 'Prediction failed!'