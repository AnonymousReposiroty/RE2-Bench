# Problem: sympy@@sympy_core_power.py@@as_content_primitive_L1725
# Module: sympy.core.power
# Function: as_content_primitive
# Line: 1725

from sympy.core.power import Pow
from sympy import sympify

def test_input(pred_input):
    obj_ins =Pow(sympify(2), sympify("1/2"))
    obj_ins_pred = sympify(pred_input['self']["__repr__"])
    assert obj_ins.as_content_primitive(radical = False, clear = False)==obj_ins_pred.as_content_primitive(radical = pred_input['args']['radical'], clear = pred_input['args']['clear']), 'Prediction failed!'
