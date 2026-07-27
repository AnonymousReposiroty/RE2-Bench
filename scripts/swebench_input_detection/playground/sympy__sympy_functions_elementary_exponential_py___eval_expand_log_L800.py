# Problem: sympy@@sympy_functions_elementary_exponential.py@@_eval_expand_log_L800
# Module: sympy.functions.elementary.exponential
# Function: _eval_expand_log
# Line: 800

from sympy.functions.elementary.exponential import log


def test_input(pred_input):
    obj_ins = log(2)
    obj_ins.nargs = '{1, 2}'
    obj_ins_pred = log(2)
    obj_ins_pred.nargs = pred_input['self']['nargs']
    assert obj_ins._eval_expand_log(deep = True, force = False, factor = False, power_base = False, power_exp = False, mul = False, log = True, multinomial = False, basic = False)==obj_ins_pred._eval_expand_log(deep = pred_input['args']['deep'], force = pred_input['kwargs']['force'], factor = pred_input['kwargs']['factor'], power_base = pred_input['kwargs']['power_base'], power_exp = pred_input['kwargs']['power_exp'], mul = pred_input['kwargs']['mul'], log = pred_input['kwargs']['log'], multinomial = pred_input['kwargs']['multinomial'], basic = pred_input['kwargs']['basic']), 'Prediction failed!'
