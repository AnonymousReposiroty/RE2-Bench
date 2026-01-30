# Problem: matplotlib@@matplotlib_path.py@@intersects_path_L667
# Module: matplotlib.path
# Function: intersects_path
# Line: 667
import numpy as np
from matplotlib.path import Path


def test_input(pred_input):
    obj_ins = Path(_interpolation_steps = 1, vertices=np.array([[0., 0.],[2., 0.]]))
    obj_ins._vertices = np.array([[0., 0.],[2., 0.]])
    obj_ins._codes = None
    obj_ins._simplify_threshold = 0.1111111111111111
    obj_ins._should_simplify = False
    obj_ins._readonly = False
    obj_ins_pred = Path(_interpolation_steps = pred_input['self']['_interpolation_steps'])
    obj_ins_pred._vertices = pred_input['self']['_vertices']
    obj_ins_pred._codes = pred_input['self']['_codes']
    obj_ins_pred._simplify_threshold = pred_input['self']['_simplify_threshold']
    obj_ins_pred._should_simplify = pred_input['self']['_should_simplify']
    obj_ins_pred._readonly = pred_input['self']['_readonly']
    assert obj_ins.intersects_path(other = Path(np.array([[ 0,0],[ 1.99969539, -0.03490481]]), None), filled = True)==obj_ins_pred.intersects_path(other = pred_input['args']['other'], filled = pred_input['args']['filled']), 'Prediction failed!'
    
    
    
obj_ins = Path(_interpolation_steps = 1, vertices=np.array([[0., 0.],[2., 0.]]))
obj_ins._vertices = np.array([[0., 0.],[2., 0.]])
obj_ins._codes = None
obj_ins._simplify_threshold = 0.1111111111111111
obj_ins._should_simplify = False
obj_ins._readonly = False