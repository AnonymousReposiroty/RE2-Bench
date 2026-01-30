# Problem: matplotlib@@matplotlib_colors.py@@_repr_html__L2140
# Module: matplotlib.colors
# Function: _repr_html_
# Line: 2140
import numpy as np
from matplotlib.colors import BivarColormap


def test_input(pred_input):
    obj_ins = BivarColormap(name = 'BiCone', N = 256, M = 256)
    obj_ins.patch = np.array([[[0.   , 0.674, 0.931],[0.   , 0.68 , 0.922],[0.   , 0.685, 0.914],[0.242, 0.69 , 0.19 ],[0.246, 0.684, 0.165],[0.25 , 0.678, 0.136]],[[0.   , 0.675, 0.934],[0.   , 0.681, 0.925],[0.   , 0.687, 0.917],[0.277, 0.692, 0.196],[0.28 , 0.686, 0.17 ],[0.283, 0.68 , 0.143]],[[0.   , 0.676, 0.937],[0.   , 0.682, 0.928],[0.   , 0.688, 0.92 ],[0.308, 0.694, 0.201],[0.31 , 0.688, 0.176],[0.312, 0.682, 0.149]],[[0.802, 0.385, 0.965],[0.808, 0.395, 0.957],[0.813, 0.405, 0.948],[0.927, 0.417, 0.167],[0.923, 0.411, 0.142],[0.919, 0.405, 0.116]],[[0.806, 0.37 , 0.963],[0.811, 0.38 , 0.954],[0.816, 0.39 , 0.945],[0.927, 0.405, 0.162],[0.923, 0.399, 0.138],[0.918, 0.393, 0.111]],[[0.809, 0.354, 0.96 ],[0.814, 0.364, 0.951],[0.819, 0.375, 0.942],[0.927, 0.393, 0.158],[0.923, 0.387, 0.134],[0.918, 0.381, 0.107]]])
    obj_ins._shape = 'circle'
    obj_ins._rgba_bad = np.array([0., 0., 0., 0.])
    obj_ins._rgba_outside = np.array([1., 0., 1., 1.])
    obj_ins._isinit = False
    obj_ins.n_variates = 2
    obj_ins._origin = [0.5, 0.5]
    obj_ins_pred = BivarColormap(name = pred_input['self']['name'], N = pred_input['self']['N'], M = pred_input['self']['M'])
    obj_ins_pred.patch = pred_input['self']['patch']
    obj_ins_pred._shape = pred_input['self']['_shape']
    obj_ins_pred._rgba_bad = pred_input['self']['_rgba_bad']
    obj_ins_pred._rgba_outside = pred_input['self']['_rgba_outside']
    obj_ins_pred._isinit = pred_input['self']['_isinit']
    obj_ins_pred.n_variates = pred_input['self']['n_variates']
    obj_ins_pred._origin = pred_input['self']['_origin']
    assert obj_ins._repr_html_()==obj_ins_pred._repr_html_(), 'Prediction failed!'
