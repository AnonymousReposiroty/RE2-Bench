# Problem: matplotlib@@matplotlib_transforms.py@@_interval_contains_close_L2948
# Module: matplotlib.transforms
# Function: _interval_contains_close
# Line: 2948
import numpy as np
from matplotlib.transforms import _interval_contains_close


def test_input(pred_input):
    assert _interval_contains_close(interval = 'array([0.001, 1.   ])', val = 'array(0.2)', rtol = 1e-10)==_interval_contains_close(interval = pred_input['args']['interval'], val = pred_input['args']['val'], rtol = pred_input['args']['rtol']), 'Prediction failed!'
    