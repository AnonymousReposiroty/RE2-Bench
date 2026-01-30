# Problem: matplotlib@@matplotlib_gridspec.py@@get_grid_positions_L145
# Module: matplotlib.gridspec
# Function: get_grid_positions
# Line: 145

from matplotlib.gridspec import GridSpecBase


def test_input(pred_input):
    obj_ins = GridSpecBase(nrows = 1, ncols = 1)
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
    obj_ins_pred = GridSpecBase()
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
    assert obj_ins.get_grid_positions(fig = '<Figure size 640x480 with 0 Axes>')==obj_ins_pred.get_grid_positions(fig = pred_input['args']['fig']), 'Prediction failed!'
    