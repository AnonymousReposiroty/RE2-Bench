# Problem: matplotlib@@matplotlib_gridspec.py@@get_topmost_subplotspec_L674
# Module: matplotlib.gridspec
# Function: get_topmost_subplotspec
# Line: 674
import numpy as np
from matplotlib.gridspec import SubplotSpec
from matplotlib.gridspec import GridSpec

def test_input(pred_input):
    obj_ins = SubplotSpec(num1 = np.int64(0), gridspec = GridSpec(1, 1))
    obj_ins._gridspec = GridSpec(1, 1)
    obj_ins._num2 = np.int64(0)
    obj_ins_pred = SubplotSpec(num1 = pred_input['self']['num1'], gridspec = GridSpec(1, 1))
    obj_ins_pred._gridspec = GridSpec(1, 1)
    obj_ins_pred._num2 = pred_input['self']['_num2']
    assert obj_ins.get_topmost_subplotspec()==obj_ins_pred.get_topmost_subplotspec(), 'Prediction failed!'
