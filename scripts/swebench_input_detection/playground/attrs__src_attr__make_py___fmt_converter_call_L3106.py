# Problem: attrs@@src_attr__make.py@@_fmt_converter_call_L3106
# Module: attr._make
# Function: _fmt_converter_call
# Line: 3106

from attr._make import Converter


def test_input(pred_input):
    obj_ins = Converter(converter=print, takes_self=False, takes_field=False)
    obj_ins_pred = pred_input['self']
    assert obj_ins._fmt_converter_call(attr_name = 'a', value_var = 'a')==obj_ins_pred._fmt_converter_call(attr_name = pred_input['args']['attr_name'], value_var = pred_input['args']['value_var']), 'Prediction failed!'