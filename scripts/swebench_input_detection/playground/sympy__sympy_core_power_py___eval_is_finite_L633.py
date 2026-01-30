# Problem: sympy@@sympy_core_power.py@@_eval_is_finite_L633
# Module: sympy.core.power
# Function: _eval_is_finite
# Line: 633

from sympy.core.power import Pow
from sympy import sympify
def test_input(pred_input):
    obj_ins = Pow(sympify("cos(phi)"), 2)
    obj_ins_pred = Pow(sympify(pred_input['self'].split("**")[0], evaluate=False), sympify(pred_input['self'].split("**")[1], evaluate=False))
    assert obj_ins._eval_is_finite()==obj_ins_pred._eval_is_finite(), 'Prediction failed!'
    
