# Problem: matplotlib@@matplotlib_cbook.py@@delete_masked_points_L939
# Module: matplotlib.cbook
# Function: delete_masked_points
# Line: 939
import numpy as np
from matplotlib.cbook import delete_masked_points


def test_input(pred_input):
    assert delete_masked_points(np.array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45]), np.array([0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]), None)==delete_masked_points(args = pred_input['args']['args']), 'Prediction failed!'