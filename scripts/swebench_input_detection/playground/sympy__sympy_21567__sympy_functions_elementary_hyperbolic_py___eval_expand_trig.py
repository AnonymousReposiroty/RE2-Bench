# Problem: sympy__sympy-21567@@sympy.functions.elementary.hyperbolic.py@@_eval_expand_trig
# Benchmark: Swebench
# Module: sympy.functions.elementary.hyperbolic
# Function: _eval_expand_trig

from sympy.functions.elementary.hyperbolic import ReciprocalHyperbolicFunction


def test_input(pred_input):
    obj_ins = ReciprocalHyperbolicFunction()
    obj_ins.nargs = {'_args_set': "{'1'}"}
    obj_ins_pred = ReciprocalHyperbolicFunction()
    obj_ins_pred.nargs = pred_input['self']['nargs']
    assert obj_ins._eval_expand_trig()==obj_ins_pred._eval_expand_trig(), 'Prediction failed!'