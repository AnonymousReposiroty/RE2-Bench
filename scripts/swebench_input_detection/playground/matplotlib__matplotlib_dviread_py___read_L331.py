# Problem: matplotlib@@matplotlib_dviread.py@@_read_L331
# Module: matplotlib.dviread
# Function: _read
# Line: 331

from matplotlib.dviread import Dvi


def test_input(pred_input):
    obj_ins = Dvi(dpi = 100.0)
    obj_ins.file = "<_io.BufferedReader name='/home/changshu/.cache/matplotlib/tex.cache/b6/1d/b61df382f15a0a6330cb8dd5ed832ae7f3ef1151ee0b944dc256b7a2b9938b84.dvi'>"
    obj_ins.fonts = {'29': "<DviFont: b'phvr7t'>"}
    obj_ins.state = '<DviState.outer: 2>'
    obj_ins._missing_font = None
    obj_ins._baseline_v = None
    obj_ins.text = [[983040, 1441792, "<DviFont: b'phvr8r'>", 50, 437251], [1420291, 1441792, "<DviFont: b'phvr8r'>", 48, 437251], [1857542, 1441792, "<DviFont: b'phvr8r'>", 37, 699130]]
    obj_ins.boxes = []
    obj_ins.f = 29
    obj_ins_pred = Dvi(dpi = pred_input['self']['dpi'])
    obj_ins_pred.file = pred_input['self']['file']
    obj_ins_pred.fonts = pred_input['self']['fonts']
    obj_ins_pred.state = pred_input['self']['state']
    obj_ins_pred._missing_font = pred_input['self']['_missing_font']
    obj_ins_pred._baseline_v = pred_input['self']['_baseline_v']
    obj_ins_pred.text = pred_input['self']['text']
    obj_ins_pred.boxes = pred_input['self']['boxes']
    obj_ins_pred.f = pred_input['self']['f']
    assert obj_ins._read()==obj_ins_pred._read(), 'Prediction failed!'