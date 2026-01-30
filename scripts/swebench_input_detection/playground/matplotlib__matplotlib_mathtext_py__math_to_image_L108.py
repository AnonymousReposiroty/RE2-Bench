# Problem: matplotlib@@matplotlib_mathtext.py@@math_to_image_L108
# Module: matplotlib.mathtext
# Function: math_to_image
# Line: 108
from pathlib import Path
from matplotlib.mathtext import math_to_image


def test_input(pred_input):
    assert math_to_image(s = '$x^2$', filename_or_obj = Path('/tmp/pytest-of-changshu/pytest-46/test_math_to_image0/example.png'), prop = None, dpi = None, format = None, color = None)==math_to_image(s = pred_input['args']['s'], filename_or_obj = Path(pred_input['args']['filename_or_obj']), prop = pred_input['args']['prop'], dpi = pred_input['args']['dpi'], format = pred_input['args']['format'], color = pred_input['args']['color']), 'Prediction failed!'
    