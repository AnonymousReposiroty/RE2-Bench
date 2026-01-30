# Problem: matplotlib@@matplotlib_tri__triangulation.py@@get_masked_triangles_L133
# Module: matplotlib.tri._triangulation
# Function: get_masked_triangles
# Line: 133

from matplotlib.tri._triangulation import Triangulation
import numpy as np

def test_input(pred_input):
    obj_ins = Triangulation(x = np.array([ 0.,  5., 10., 15., 20.,  1.,  6., 11., 16., 21.,  2.,  7., 12.,
       17., 22.,  3.,  8., 13., 18., 23.,  4.,  9., 14., 19., 24.]), y = np.array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]), mask = None, triangles = np.array([[ 0,  1,  5],
       [ 8, 13, 12],
       [ 1,  2,  6],
       [10,  5,  6],
       [ 6,  5,  1],
       [17, 16, 12],
       [12, 13, 17],
       [20, 16, 21],
       [21, 17, 22],
       [16, 17, 21],
       [15, 16, 20],
       [23, 19, 24],
       [14, 19, 18],
       [18, 13, 14],
       [18, 17, 13],
       [19, 23, 18],
       [18, 23, 22],
       [22, 17, 18],
       [ 4,  8,  3],
       [ 9,  8,  4],
       [14, 13,  9],
       [13,  8,  9],
       [12, 16, 11],
       [10,  6, 11],
       [11, 15, 10],
       [16, 15, 11],
       [ 7,  8, 12],
       [ 7,  6,  2],
       [12, 11,  7],
       [ 7, 11,  6],
       [ 2,  3,  7],
       [ 7,  3,  8]], dtype=np.int32))
    obj_ins._edges = None
    obj_ins._neighbors = None
    obj_ins.is_delaunay = True
    obj_ins._cpp_triangulation = None
    obj_ins._trifinder = None
    obj_ins_pred = Triangulation(x = pred_input['self']['x'], y = pred_input['self']['y'], mask = pred_input['self']['mask'], triangles = pred_input['self']['triangles'])
    obj_ins_pred._edges = pred_input['self']['_edges']
    obj_ins_pred._neighbors = pred_input['self']['_neighbors']
    obj_ins_pred.is_delaunay = pred_input['self']['is_delaunay']
    obj_ins_pred._cpp_triangulation = pred_input['self']['_cpp_triangulation']
    obj_ins_pred._trifinder = pred_input['self']['_trifinder']
    assert obj_ins.get_masked_triangles()==obj_ins_pred.get_masked_triangles(), 'Prediction failed!'
