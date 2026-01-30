# Problem: matplotlib@@matplotlib_gridspec.py@@subplots_L250
# Module: matplotlib.gridspec
# Function: GridSpec.subplots (defined in GridSpecBase)
# Line: 250

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def test_input(pred_input):
    """
    Test case for GridSpec.subplots method.

    Ground truth input:
    - self: GridSpec with nrows=1, ncols=1, attached to a figure
    - sharex = False
    - sharey = False
    - squeeze = True
    - subplot_kw = None

    Expected: returns a single Axes object (since 1x1 grid with squeeze=True)
    """
    # Create ground truth GridSpec with figure
    fig_gt = plt.figure()
    gs_gt = GridSpec(
        nrows=1,
        ncols=1,
        figure=fig_gt,
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=None,
        hspace=None,
        height_ratios=[1],
        width_ratios=[1]
    )

    # Call with ground truth input
    result_gt = gs_gt.subplots(sharex=False, sharey=False, squeeze=True, subplot_kw=None)

    # Create predicted GridSpec with figure
    fig_pred = plt.figure()
    gs_pred = GridSpec(
        nrows=pred_input['self']['_nrows'],
        ncols=pred_input['self']['_ncols'],
        figure=fig_pred,
        left=pred_input['self']['left'],
        bottom=pred_input['self']['bottom'],
        right=pred_input['self']['right'],
        top=pred_input['self']['top'],
        wspace=pred_input['self']['wspace'],
        hspace=pred_input['self']['hspace'],
        height_ratios=pred_input['self']['_row_height_ratios'],
        width_ratios=pred_input['self']['_col_width_ratios']
    )

    # Call with predicted input
    result_pred = gs_pred.subplots(
        sharex=pred_input['args']['sharex'],
        sharey=pred_input['args']['sharey'],
        squeeze=pred_input['args']['squeeze'],
        subplot_kw=pred_input['args']['subplot_kw']
    )

    # Compare results - both should be single Axes objects for 1x1 grid with squeeze=True
    assert type(result_gt) == type(result_pred), \
        f'Type mismatch! Expected {type(result_gt)}, got {type(result_pred)}'

    # For Axes, compare that they have same structure
    if hasattr(result_gt, 'shape'):
        assert result_gt.shape == result_pred.shape, \
            f'Shape mismatch! Expected {result_gt.shape}, got {result_pred.shape}'

    plt.close('all')
    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for GridSpec.subplots method output prediction.

    Ground truth:
    - GridSpec: 1x1, attached to figure
    - sharex=False, sharey=False, squeeze=True, subplot_kw=None
    - return: single Axes object
    """
    # Create ground truth GridSpec with figure
    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=1, figure=fig)

    # Call with ground truth input
    result_gt = gs.subplots(sharex=False, sharey=False, squeeze=True, subplot_kw=None)

    # Compare types
    assert type(result_gt).__name__ == type(pred_output).__name__, \
        f'Type mismatch! Expected {type(result_gt).__name__}, got {type(pred_output).__name__}'

    plt.close('all')
    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'left': None,
            'bottom': None,
            'right': None,
            'top': None,
            'wspace': None,
            'hspace': None,
            'figure': '<Figure size 640x480 with 0 Axes>',
            '_nrows': 1,
            '_ncols': 1,
            '_row_height_ratios': [1],
            '_col_width_ratios': [1]
        },
        'args': {
            'sharex': False,
            'sharey': False,
            'squeeze': True,
            'subplot_kw': None
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=1, figure=fig)
    gt_output = gs.subplots(sharex=False, sharey=False, squeeze=True, subplot_kw=None)
    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output: {gt_output}")

    test_output(gt_output)

    plt.close('all')
