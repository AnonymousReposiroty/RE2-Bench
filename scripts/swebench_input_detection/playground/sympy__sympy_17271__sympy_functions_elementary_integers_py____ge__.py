# Problem: sympy__sympy-17271@@sympy.functions.elementary.integers.py@@__ge__
# Benchmark: Swebench
# Module: sympy.functions.elementary.integers
# Function: __ge__

from sympy.functions.elementary.integers import ceiling


def test_input(pred_input):
    obj_ins = ceiling('x')
    obj_ins.nargs = {}
    obj_ins_pred = ceiling(pred_input['self']['nargs'])
    obj_ins_pred.nargs = pred_input['self']['nargs']
    assert obj_ins.__ge__(other = 'x')==obj_ins_pred.__ge__(other = pred_input['args']['other']), 'Prediction failed!'
