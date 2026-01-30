# Problem: matplotlib@@matplotlib_colors.py@@__call___L755
# Module: matplotlib.colors
# Function: __call__
# Line: 755

from matplotlib.colors import Colormap
import numpy as np

def test_input(pred_input):
    obj_ins = Colormap(name = 'Reds', N = 256)
    obj_ins.monochrome = False
    obj_ins._rgba_bad = [0.0, 0.0, 0.0, 0.0]
    obj_ins._rgba_under = None
    obj_ins._rgba_over = None
    obj_ins._i_under = 256
    obj_ins._i_over = 257
    obj_ins._i_bad = 258
    obj_ins._isinit = False
    obj_ins.n_variates = 1
    obj_ins.colorbar_extend = False
    obj_ins._segmentdata = {'red': np.array([[0.        , 1.        , 1.        ], [0.125     , 0.99607843, 0.99607843], [0.25      , 0.98823529, 0.98823529], [0.375     , 0.98823529, 0.98823529], [0.5       , 0.98431373, 0.98431373], [0.625     , 0.9372549 , 0.9372549 ], [0.75      , 0.79607843, 0.79607843], [0.875     , 0.64705882, 0.64705882], [1.        , 0.40392157, 0.40392157]]), 'green': np.array([[0.        , 0.96078431, 0.96078431], [0.125     , 0.87843137, 0.87843137], [0.25      , 0.73333333, 0.73333333], [0.375     , 0.57254902, 0.57254902], [0.5       , 0.41568627, 0.41568627], [0.625     , 0.23137255, 0.23137255], [0.75      , 0.09411765, 0.09411765], [0.875     , 0.05882353, 0.05882353], [1.        , 0.        , 0.        ]]), 'blue': np.array([[0.        , 0.94117647, 0.94117647], [0.125     , 0.82352941, 0.82352941], [0.25      , 0.63137255, 0.63137255], [0.375     , 0.44705882, 0.44705882], [0.5       , 0.29019608, 0.29019608], [0.625     , 0.17254902, 0.17254902], [0.75      , 0.11372549, 0.11372549], [0.875     , 0.08235294, 0.08235294], [1.        , 0.05098039, 0.05098039]]), 'alpha': np.array([[0.   , 1.   , 1.   ], [0.125, 1.   , 1.   ], [0.25 , 1.   , 1.   ], [0.375, 1.   , 1.   ], [0.5  , 1.   , 1.   ], [0.625, 1.   , 1.   ],[0.75 , 1.   , 1.   ],[0.875, 1.   , 1.   ],[1.   , 1.   , 1.   ]])}
    obj_ins._gamma = 1.0
    obj_ins_pred = Colormap(name = pred_input['self']['name'], N = pred_input['self']['N'])
    obj_ins_pred.monochrome = pred_input['self']['monochrome']
    obj_ins_pred._rgba_bad = pred_input['self']['_rgba_bad']
    obj_ins_pred._rgba_under = pred_input['self']['_rgba_under']
    obj_ins_pred._rgba_over = pred_input['self']['_rgba_over']
    obj_ins_pred._i_under = pred_input['self']['_i_under']
    obj_ins_pred._i_over = pred_input['self']['_i_over']
    obj_ins_pred._i_bad = pred_input['self']['_i_bad']
    obj_ins_pred._isinit = pred_input['self']['_isinit']
    obj_ins_pred.n_variates = pred_input['self']['n_variates']
    obj_ins_pred.colorbar_extend = pred_input['self']['colorbar_extend']
    obj_ins_pred._segmentdata = pred_input['self']['_segmentdata']
    obj_ins_pred._gamma = pred_input['self']['_gamma']
    data = np.array([
    [0.57647059, 0.56470588, 0.52941176, 0.47058824, 0.38823529, 0.28235294, 0.15294118, 0.        ],
    [0.58823529, 0.57647059, 0.54117647, 0.48235294, 0.4       , 0.29411765, 0.16470588, 0.01176471],
    [0.62352941, 0.61176471, 0.57647059, 0.51764706, 0.43529412, 0.32941176, 0.2       , 0.04705882],
    [0.68235294, 0.67058824, 0.63529412, 0.57647059, 0.49411765, 0.38823529, 0.25882353, 0.10588235],
    [0.76470588, 0.75294118, 0.71764706, 0.65882353, 0.57647059, 0.47058824, 0.34117647, 0.18823529],
    [0.87058824, 0.85882353, 0.82352941, 0.76470588, 0.68235294, 0.57647059, 0.44705882, 0.29411765],
    [1.        , 0.98823529, 0.95294118, 0.89411765, 0.81176471, 0.70588235, 0.57647059, 0.42352941],
], dtype=float)

    m = np.ma.masked_array(data, mask=False, fill_value=1e20)
    assert obj_ins.__call__(X = m, alpha = None, bytes = False)==obj_ins_pred.__call__(X = pred_input['args']['X'], alpha = pred_input['args']['alpha'], bytes = pred_input['args']['bytes']), 'Prediction failed!'
