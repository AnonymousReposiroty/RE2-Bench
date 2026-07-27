# Problem: matplotlib@@matplotlib_path.py@@copy_L271
# Module: matplotlib.path
# Function: copy
# Line: 271
import numpy as np
from matplotlib.path import Path


def test_input(pred_input):
    obj_ins = Path(_interpolation_steps = 1, vertices=np.array([[0., 0.],[1., 1.]]))
    obj_ins._vertices = np.array([[0., 0.],[1., 1.]])
    obj_ins._codes = np.array([1, 2], dtype=np.uint8)
    obj_ins._simplify_threshold = 0.1111111111111111
    obj_ins._should_simplify = False
    obj_ins._readonly = False
    obj_ins_pred = Path(_interpolation_steps = pred_input['self']['_interpolation_steps'], vertices=pred_input['self']['_vertices'])
    obj_ins_pred._vertices = pred_input['self']['_vertices']
    obj_ins_pred._codes = pred_input['self']['_codes']
    obj_ins_pred._simplify_threshold = pred_input['self']['_simplify_threshold']
    obj_ins_pred._should_simplify = pred_input['self']['_should_simplify']
    obj_ins_pred._readonly = pred_input['self']['_readonly']
    assert obj_ins.copy()==obj_ins_pred.copy(), 'Prediction failed!'
    
    