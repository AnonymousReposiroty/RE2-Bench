# Problem: sympy__sympy-17770@@sympy.functions.elementary.hyperbolic.py@@_eval_is_finite
# Benchmark: Swebench
# Module: sympy.functions.elementary.hyperbolic
# Function: _eval_is_finite

from sympy.functions.elementary.hyperbolic import sinh


def test_input(pred_input):
    obj_ins = sinh('x')
    obj_ins.nargs = {}
    obj_ins_pred = sinh('x')
    obj_ins_pred.nargs = pred_input['self']['nargs']
    assert obj_ins._eval_is_finite()==obj_ins_pred._eval_is_finite(), 'Prediction failed!'
