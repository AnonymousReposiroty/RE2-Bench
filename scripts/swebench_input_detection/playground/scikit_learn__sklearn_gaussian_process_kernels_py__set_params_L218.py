# Problem: scikit_learn__sklearn_gaussian_process_kernels_py__set_params_L218
# Module: sklearn.gaussian_process.kernels
# Function: set_params
# Line: 218

import numpy as np
from sklearn.gaussian_process.kernels import RBF


def create_kernel_gt():
    """
    Create the ground truth RBF kernel with specified parameters.
    """
    return RBF(length_scale=0.10000000000000002, length_scale_bounds=(1e-05, 100000.0))


def test_input(pred_input):
    # Create ground truth self kernel
    kernel_gt = create_kernel_gt()

    # Ground truth kwargs
    kwargs_gt = {
        "length_scale": 0.11197917226857704,
        "length_scale_bounds": (1e-05, 100000.0)
    }

    # Call set_params on ground truth
    result_gt = kernel_gt.set_params(**kwargs_gt)

    # Store ground truth state after set_params
    gt_length_scale = kernel_gt.length_scale
    gt_length_scale_bounds = kernel_gt.length_scale_bounds

    # Create predicted self kernel from input
    length_scale = pred_input['self'].get('length_scale', 1.0)
    length_scale_bounds = pred_input['self'].get('length_scale_bounds', (1e-5, 1e5))
    if isinstance(length_scale_bounds, list):
        length_scale_bounds = tuple(length_scale_bounds)

    kernel_pred = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    # Parse predicted kwargs from input
    kwargs_pred = {}
    for key, value in pred_input.get('kwargs', {}).items():
        if key == 'length_scale_bounds' and isinstance(value, list):
            kwargs_pred[key] = tuple(value)
        else:
            kwargs_pred[key] = value

    # Call set_params on predicted
    result_pred = kernel_pred.set_params(**kwargs_pred)

    # Compare results - set_params returns self, so compare the kernel state
    assert result_gt is kernel_gt, 'GT set_params should return self'
    assert result_pred is kernel_pred, 'Pred set_params should return self'

    # Compare the kernel parameters after set_params
    assert np.isclose(kernel_gt.length_scale, kernel_pred.length_scale), \
        f'length_scale mismatch! GT: {kernel_gt.length_scale}, Pred: {kernel_pred.length_scale}'

    gt_bounds = kernel_gt.length_scale_bounds
    pred_bounds = kernel_pred.length_scale_bounds
    if isinstance(gt_bounds, tuple) and isinstance(pred_bounds, tuple):
        assert np.allclose(gt_bounds, pred_bounds), \
            f'length_scale_bounds mismatch! GT: {gt_bounds}, Pred: {pred_bounds}'
    else:
        assert gt_bounds == pred_bounds, \
            f'length_scale_bounds mismatch! GT: {gt_bounds}, Pred: {pred_bounds}'

    print('Test passed!')


if __name__ == '__main__':
    # Test with ground truth input
    gt_input = {
        "self": {
            "length_scale": 0.10000000000000002,
            "length_scale_bounds": [1e-05, 100000.0]
        },
        "args": {},
        "kwargs": {
            "length_scale": 0.11197917226857704,
            "length_scale_bounds": [1e-05, 100000.0]
        }
    }
    test_input(gt_input)
