# Problem: Pillow@@src_PIL_PdfParser.py@@__eq___L96
# Module: PIL.PdfParser
# Function: __eq__
# Line: 96

from PIL.PdfParser import IndirectReference


def test_input(pred_input):
    obj_ins = IndirectReference(1,2)
    obj_ins_pred = IndirectReference(1,2)
    assert obj_ins.__eq__(other = [1, 2])==obj_ins_pred.__eq__(other = pred_input['args']['other']), 'Prediction failed!'
