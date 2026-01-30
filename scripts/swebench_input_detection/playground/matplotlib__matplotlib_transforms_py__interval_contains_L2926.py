# Problem: matplotlib@@matplotlib_transforms.py@@interval_contains_L2926
# Module: matplotlib.transforms
# Function: interval_contains
# Line: 2926

from matplotlib.transforms import interval_contains
import numpy as np

def test_input(pred_input):
    assert interval_contains(interval = np.array([-107.73502692,   -7.73502692]), val = -80.0)==interval_contains(interval = pred_input['args']['interval'], val = pred_input['args']['val']), 'Prediction failed!'
