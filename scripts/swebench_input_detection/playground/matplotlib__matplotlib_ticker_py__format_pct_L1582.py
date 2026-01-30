# Problem: matplotlib@@matplotlib_ticker.py@@format_pct_L1582
# Module: matplotlib.ticker
# Function: format_pct
# Line: 1582

from matplotlib.ticker import PercentFormatter


def test_input(pred_input):
    obj_ins = PercentFormatter(xmax = 100.0, decimals = 0)
    obj_ins._symbol = '%'
    obj_ins._is_latex = False
    obj_ins_pred = PercentFormatter(xmax = pred_input['self']['xmax'], decimals = pred_input['self']['decimals'])
    obj_ins_pred._symbol = pred_input['self']['_symbol']
    obj_ins_pred._is_latex = pred_input['self']['_is_latex']
    assert obj_ins.format_pct(x = 120, display_range = 100)==obj_ins_pred.format_pct(x = pred_input['args']['x'], display_range = pred_input['args']['display_range']), 'Prediction failed!'
    
