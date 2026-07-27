# Problem: sympy__sympy-13678@@sympy.functions.elementary.hyperbolic.py@@_eval_is_real
# Benchmark: Swebench
# Module: sympy.functions.elementary.hyperbolic
# Function: _eval_is_real

from sympy.functions.elementary.hyperbolic import sinh

from sympy import I
def test_input(pred_input):
    obj_ins = sinh(I, evaluate=False)
    obj_ins.nargs = {'_elements': 'frozenset({1})'}
    obj_ins_pred = sinh(I, evaluate=False)
    obj_ins_pred.nargs = pred_input['self']['nargs']
    assert obj_ins._eval_is_real()==obj_ins_pred._eval_is_real(), 'Prediction failed!'
