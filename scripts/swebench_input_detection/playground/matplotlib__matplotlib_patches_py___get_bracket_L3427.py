# Problem: matplotlib@@matplotlib_patches.py@@_get_bracket_L3427
# Module: matplotlib.patches
# Function: _get_bracket
# Line: 3427

from matplotlib.patches import _Curve


def test_input(pred_input):
    obj_ins = _Curve(head_length = 0.4, head_width = 0.2, widthA = 1.0, widthB = 1.0, lengthA = 0.2, lengthB = 0.2, angleA = 0, angleB = None, scaleA = None, scaleB = None)
    obj_ins._beginarrow_head = True
    obj_ins._beginarrow_bracket = False
    obj_ins._endarrow_head = False
    obj_ins._endarrow_bracket = True
    obj_ins_pred = _Curve(head_length = pred_input['self']['head_length'], head_width = pred_input['self']['head_width'], widthA = pred_input['self']['widthA'], widthB = pred_input['self']['widthB'], lengthA = pred_input['self']['lengthA'], lengthB = pred_input['self']['lengthB'], angleA = pred_input['self']['angleA'], angleB = pred_input['self']['angleB'], scaleA = pred_input['self']['scaleA'], scaleB = pred_input['self']['scaleB'])
    obj_ins_pred._beginarrow_head = pred_input['self']['_beginarrow_head']
    obj_ins_pred._beginarrow_bracket = pred_input['self']['_beginarrow_bracket']
    obj_ins_pred._endarrow_head = pred_input['self']['_endarrow_head']
    obj_ins_pred._endarrow_bracket = pred_input['self']['_endarrow_bracket']
    assert obj_ins._get_bracket(x0 = 358.0020718742162, y0 = 329.4117647058823, x1 = 220.00292802695185, y1 = 329.4117647058823, width = 25.0, length = 5.0, angle = None)==obj_ins_pred._get_bracket(x0 = pred_input['args']['x0'], y0 = pred_input['args']['y0'], x1 = pred_input['args']['x1'], y1 = pred_input['args']['y1'], width = pred_input['args']['width'], length = pred_input['args']['length'], angle = pred_input['args']['angle']), 'Prediction failed!'
    