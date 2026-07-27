# Problem: matplotlib@@matplotlib_cbook.py@@pts_to_midstep_L1652
# Module: matplotlib.cbook
# Function: pts_to_midstep
# Line: 1652
import numpy as np
from matplotlib.cbook import pts_to_midstep


def test_input(pred_input):
    assert pts_to_midstep(np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))==pts_to_midstep(x = pred_input['args']['x'], args = pred_input['args']['args']), 'Prediction failed!'


