# Problem: matplotlib@@matplotlib_bezier.py@@axis_aligned_extrema_L288
# Module: matplotlib.bezier
# Function: axis_aligned_extrema
# Line: 288

from matplotlib.bezier import BezierSegment
import numpy as np

def test_input(pred_input):
    obj_ins = BezierSegment(control_points=np.array([[ 0.        , -2.48451997]]))
    obj_ins._cpoints = np.array([[ 0.        , -2.48451997]])
    obj_ins._N = 1
    obj_ins._d = 2
    obj_ins._orders = np.array([0])
    obj_ins._px = np.array([[ 0.        , -2.48451997]])
    obj_ins_pred = BezierSegment(control_points=np.array([[ 0.        , -2.48451997]]))
    obj_ins_pred._cpoints = pred_input['self']['_cpoints']
    obj_ins_pred._N = pred_input['self']['_N']
    obj_ins_pred._d = pred_input['self']['_d']
    obj_ins_pred._orders = pred_input['self']['_orders']
    obj_ins_pred._px = pred_input['self']['_px']
    assert obj_ins.axis_aligned_extrema()==obj_ins_pred.axis_aligned_extrema(), 'Prediction failed!'