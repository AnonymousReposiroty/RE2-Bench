# Problem: scikit_learn__sklearn_gaussian_process_kernels_py____eq___L382
# Module: sklearn.gaussian_process.kernels
# Function: __eq__
# Line: 382

import numpy as np
from sklearn.gaussian_process.kernels import RBF


def create_kernel_gt():
    """
    Create the ground truth RBF kernel with specified parameters.
    """
    return RBF(length_scale=1.0, length_scale_bounds=(0.5, 2.0))


def create_kernel_b_gt():
    """
    Create the ground truth 'b' kernel for comparison.
    RBF(length_scale=1) means length_scale=1 with default bounds.
    """
    return RBF(length_scale=1)


def test_input(pred_input):
    # Create ground truth self kernel
    kernel_gt = create_kernel_gt()

    # Create ground truth b kernel
    b_gt = create_kernel_b_gt()

    # Call __eq__ on ground truth
    result_gt = kernel_gt.__eq__(b_gt)

    # Create predicted self kernel from input
    length_scale = pred_input['self'].get('length_scale', 1.0)
    length_scale_bounds = pred_input['self'].get('length_scale_bounds', (1e-5, 1e5))
    if isinstance(length_scale_bounds, list):
        length_scale_bounds = tuple(length_scale_bounds)

    kernel_pred = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    # Parse predicted b kernel from input
    b_str = pred_input['args']['b']
    if isinstance(b_str, str):
        # Parse "RBF(length_scale=1)" format
        # Extract parameters from the string representation
        if 'RBF' in b_str:
            # Simple parsing for RBF(length_scale=X) format
            import re
            match = re.search(r'length_scale=([0-9.]+)', b_str)
            if match:
                b_length_scale = float(match.group(1))
                b_pred = RBF(length_scale=b_length_scale)
            else:
                # Default RBF
                b_pred = RBF()
        else:
            b_pred = RBF()
    else:
        # If b is already a dict with kernel params
        b_pred = RBF(**b_str) if isinstance(b_str, dict) else RBF()

    # Call __eq__ on predicted
    result_pred = kernel_pred.__eq__(b_pred)

    # Compare results
    assert result_gt == result_pred, f'Result mismatch! GT: {result_gt}, Pred: {result_pred}'

    print('Test passed!')


if __name__ == '__main__':
    # Test with ground truth input
    gt_input = {
        "self": {
            "length_scale": 1.0,
            "length_scale_bounds": [0.5, 2.0]
        },
        "args": {
            "b": "RBF(length_scale=1)"
        },
        "kwargs": {}
    }
    test_input(gt_input)
