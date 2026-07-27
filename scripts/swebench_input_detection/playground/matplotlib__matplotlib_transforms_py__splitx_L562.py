# Problem: matplotlib@@matplotlib_transforms.py@@splitx_L562
# Module: matplotlib.transforms
# Function: splitx
# Line: 562

from matplotlib.transforms import BboxBase
import numpy as np

def test_input(pred_input):
    obj_ins = BboxBase()
    obj_ins._parents = {}
    obj_ins._invalid = 2
    obj_ins._shorthand_name = ''
    obj_ins._points = np.array([[0.125, 0.1  ],[0.9  , 0.9  ]])
    obj_ins._minpos = np.array([np.inf, np.inf])
    obj_ins._ignore = True
    obj_ins._points_orig = np.array([[0.125, 0.1  ],[0.9  , 0.9  ]])
    obj_ins_pred = BboxBase()
    obj_ins_pred._parents = pred_input['self']['_parents']
    obj_ins_pred._invalid = pred_input['self']['_invalid']
    obj_ins_pred._shorthand_name = pred_input['self']['_shorthand_name']
    obj_ins_pred._points = pred_input['self']['_points']
    obj_ins_pred._minpos = pred_input['self']['_minpos']
    obj_ins_pred._ignore = pred_input['self']['_ignore']
    obj_ins_pred._points_orig = pred_input['self']['_points_orig']
    assert obj_ins.splitx(args = [0.85, 0.85])==obj_ins_pred.splitx(args = pred_input['args']['args']), 'Prediction failed!'
    