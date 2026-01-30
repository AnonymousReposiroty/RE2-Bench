# Problem: Pillow@@src_PIL_Image.py@@getmodetype_L271
# Module: PIL.Image
# Function: getmodetype
# Line: 271

from PIL.Image import getmodetype


def test_input(pred_input):
    assert getmodetype(mode = 'RGB')==getmodetype(mode = pred_input['args']['mode']), 'Prediction failed!'
