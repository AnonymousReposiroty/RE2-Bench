# Problem: Pillow@@src_PIL_ImageCms.py@@get_display_profile_L344
# Module: PIL.ImageCms
# Function: get_display_profile
# Line: 344

from PIL.ImageCms import get_display_profile


def test_input(pred_input):
    assert get_display_profile(handle = None)==get_display_profile(handle = pred_input['args']['handle']), 'Prediction failed!'
