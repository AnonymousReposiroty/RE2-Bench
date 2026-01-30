# Problem: matplotlib@@matplotlib_ticker.py@@format_data_L689
# Module: matplotlib.ticker
# Function: format_data
# Line: 689

from matplotlib.ticker import ScalarFormatter


def test_input(pred_input):
    obj_ins = ScalarFormatter()
    obj_ins._offset_threshold = 2
    obj_ins.offset = 0
    obj_ins._useOffset = True
    obj_ins._usetex = False
    obj_ins._useMathText = False
    obj_ins.orderOfMagnitude = 0
    obj_ins.format = ''
    obj_ins._scientific = True
    obj_ins._powerlimits = [-7, 7]
    obj_ins._useLocale = False
    obj_ins_pred = ScalarFormatter()
    obj_ins_pred._offset_threshold = pred_input['self']['_offset_threshold']
    obj_ins_pred.offset = pred_input['self']['offset']
    obj_ins_pred._useOffset = pred_input['self']['_useOffset']
    obj_ins_pred._usetex = pred_input['self']['_usetex']
    obj_ins_pred._useMathText = pred_input['self']['_useMathText']
    obj_ins_pred.orderOfMagnitude = pred_input['self']['orderOfMagnitude']
    obj_ins_pred.format = pred_input['self']['format']
    obj_ins_pred._scientific = pred_input['self']['_scientific']
    obj_ins_pred._powerlimits = pred_input['self']['_powerlimits']
    obj_ins_pred._useLocale = pred_input['self']['_useLocale']
    assert obj_ins.format_data(value = 0.11)==obj_ins_pred.format_data(value = pred_input['args']['value']), 'Prediction failed!'