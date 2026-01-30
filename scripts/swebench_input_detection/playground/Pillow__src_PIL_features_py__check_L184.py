# Problem: Pillow@@src_PIL_features.py@@check_L184
# Module: PIL.features
# Function: check
# Line: 184

from PIL.features import check


def test_input(pred_input):
    assert check(feature = 'freetype2')==check(feature = pred_input['args']['feature']), 'Prediction failed!'
