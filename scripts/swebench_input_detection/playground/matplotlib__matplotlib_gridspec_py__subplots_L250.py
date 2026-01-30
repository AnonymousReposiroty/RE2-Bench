# Problem: matplotlib@@matplotlib_gridspec.py@@subplots_L250
# Module: matplotlib.gridspec
# Function: subplots
# Line: 250

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6.4, 4.8))
def test_input(pred_input):
    obj_ins = GridSpec(ncols=1, nrows=1, figure=fig)
    obj_ins.left = None
    obj_ins.bottom = None
    obj_ins.right = None
    obj_ins.top = None
    obj_ins.wspace = None
    obj_ins.hspace = None
    obj_ins.figure = '<Figure size 640x480 with 0 Axes>'
    obj_ins._nrows = 1
    obj_ins._ncols = 1
    obj_ins._row_height_ratios = [1]
    obj_ins._col_width_ratios = [1]
    obj_ins_pred = GridSpec(ncols=1, nrows=1, figure=fig)
    obj_ins_pred.left = pred_input['self']['left']
    obj_ins_pred.bottom = pred_input['self']['bottom']
    obj_ins_pred.right = pred_input['self']['right']
    obj_ins_pred.top = pred_input['self']['top']
    obj_ins_pred.wspace = pred_input['self']['wspace']
    obj_ins_pred.hspace = pred_input['self']['hspace']
    obj_ins_pred.figure = pred_input['self']['figure']
    obj_ins_pred._nrows = pred_input['self']['_nrows']
    obj_ins_pred._ncols = pred_input['self']['_ncols']
    obj_ins_pred._row_height_ratios = pred_input['self']['_row_height_ratios']
    obj_ins_pred._col_width_ratios = pred_input['self']['_col_width_ratios']
    assert obj_ins.subplots(sharex = False, sharey = False, squeeze = True, subplot_kw = None)==obj_ins_pred.subplots(sharex = pred_input['args']['sharex'], sharey = pred_input['args']['sharey'], squeeze = pred_input['args']['squeeze'], subplot_kw = pred_input['args']['subplot_kw']), 'Prediction failed!'
