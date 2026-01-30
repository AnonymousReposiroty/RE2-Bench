# Problem: matplotlib@@matplotlib_font_manager.py@@get_family_L721
# Module: matplotlib.font.manager
# Function: get_family
# Line: 721

from matplotlib.font_manager import FontProperties


def test_input(pred_input):
    obj_ins = FontProperties()
    obj_ins._family = ['Helvetica']
    obj_ins._slant = 'normal'
    obj_ins._variant = 'normal'
    obj_ins._weight = 'normal'
    obj_ins._stretch = 'normal'
    obj_ins._file = None
    obj_ins._size = 12.0
    obj_ins._math_fontfamily = 'cm'
    obj_ins_pred = FontProperties()
    obj_ins_pred._family = pred_input['self']['_family']
    obj_ins_pred._slant = pred_input['self']['_slant']
    obj_ins_pred._variant = pred_input['self']['_variant']
    obj_ins_pred._weight = pred_input['self']['_weight']
    obj_ins_pred._stretch = pred_input['self']['_stretch']
    obj_ins_pred._file = pred_input['self']['_file']
    obj_ins_pred._size = pred_input['self']['_size']
    obj_ins_pred._math_fontfamily = pred_input['self']['_math_fontfamily']
    assert obj_ins.get_family()==obj_ins_pred.get_family(), 'Prediction failed!'