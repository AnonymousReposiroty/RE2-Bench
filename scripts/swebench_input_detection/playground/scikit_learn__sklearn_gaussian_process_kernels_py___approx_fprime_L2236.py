import numpy as np


# Problem: scikit-learn@@sklearn_gaussian_process_kernels.py@@_approx_fprime_L2236
# Module: sklearn.gaussian.process.kernels
# Function: _approx_fprime
# Line: 2236

from sklearn.gaussian_process.kernels import _approx_fprime


def test_input(pred_input):
    assert np.array([0.69314718]) ==pred_input['args']['xk'], 'Prediction failed!'