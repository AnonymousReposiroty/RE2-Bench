# Problem: matplotlib@@matplotlib_transforms.py@@update_from_data_xy_L965
# Module: matplotlib.transforms
# Function: update_from_data_xy
# Line: 965
import numpy as np
from matplotlib.transforms import Bbox

inf = np.inf
def test_input(pred_input):
    obj_ins = Bbox(points=np.array([[inf, inf],[-inf, -inf]]))
    obj_ins._parents = {}
    obj_ins._invalid = 0
    obj_ins._shorthand_name = ''
    obj_ins._points = np.array([[ inf,  inf],[-inf, -inf]])
    obj_ins._minpos = np.array([inf, inf])
    obj_ins._ignore = True
    obj_ins._points_orig = np.array([[ inf,  inf],[-inf, -inf]])
    obj_ins_pred = Bbox()
    obj_ins_pred._parents = pred_input['self']['_parents']
    obj_ins_pred._invalid = pred_input['self']['_invalid']
    obj_ins_pred._shorthand_name = pred_input['self']['_shorthand_name']
    obj_ins_pred._points = pred_input['self']['_points']
    obj_ins_pred._minpos = pred_input['self']['_minpos']
    obj_ins_pred._ignore = pred_input['self']['_ignore']
    obj_ins_pred._points_orig = pred_input['self']['_points_orig']
    assert obj_ins.update_from_data_xy(xy = np.array([[1., 1.],[1., 1.],[1., 1.]]), ignore = True, updatex = True, updatey = True)==obj_ins_pred.update_from_data_xy(xy = pred_input['args']['xy'], ignore = pred_input['args']['ignore'], updatex = pred_input['args']['updatex'], updatey = pred_input['args']['updatey']), 'Prediction failed!'
