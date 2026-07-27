# Problem: Pillow@@src_PIL_ImageMorph.py@@build_lut_L142
# Module: PIL.ImageMorph
# Function: build_lut
# Line: 142

from PIL.ImageMorph import LutBuilder


def test_input(pred_input):
    obj_ins = LutBuilder(patterns = ['4:(... .1. .0.)->0', '4:(... .1. ..0)->0'])
    obj_ins.lut = None
    obj_ins_pred = LutBuilder(patterns = pred_input['self']['patterns'])
    obj_ins_pred.lut = pred_input['self']['lut']
    assert obj_ins.build_lut()==obj_ins_pred.build_lut(), 'Prediction failed!'
