# Problem: matplotlib@@matplotlib_cbook.py@@_reshape_2D_L1391
# Module: matplotlib.cbook
# Function: _reshape_2D
# Line: 1391
import numpy as np
from matplotlib.cbook import _reshape_2D


def test_input(pred_input):
    assert _reshape_2D(X = np.array([[0.97291764, 0.11094361, 0.38826409],[0.78306588, 0.97289726, 0.48320961],[0.33642111, 0.56741904, 0.04794151],[0.38893703, 0.90630365, 0.16101821],[0.74362113, 0.63297416, 0.32418002]]), name = 'X')==_reshape_2D(X = pred_input['args']['X'], name = pred_input['args']['name']), 'Prediction failed!'
