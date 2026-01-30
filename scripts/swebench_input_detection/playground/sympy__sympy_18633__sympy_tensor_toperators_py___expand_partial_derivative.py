# Problem: sympy__sympy-18633@@sympy.tensor.toperators.py@@_expand_partial_derivative
# Benchmark: Swebench
# Module: sympy.tensor.toperators
# Function: _expand_partial_derivative

from sympy.tensor.toperators import PartialDerivative
from sympy import sympify

def test_input(pred_input):
    obj_ins = PartialDerivative(expr=sympify('x'))
    obj_ins._indices = None
    obj_ins._free = None
    obj_ins._dum = None
    obj_ins_pred = PartialDerivative(expr=sympify('x'))
    obj_ins_pred._indices = pred_input['self']['_indices']
    obj_ins_pred._free = pred_input['self']['_free']
    obj_ins_pred._dum = pred_input['self']['_dum']
    assert obj_ins._expand_partial_derivative()==obj_ins_pred._expand_partial_derivative(), 'Prediction failed!'
    