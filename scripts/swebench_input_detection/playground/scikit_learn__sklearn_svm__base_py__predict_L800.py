import numpy as np


# Problem: scikit-learn@@sklearn_svm__base.py@@predict_L800
# Module: sklearn.svm._base
# Function: predict
# Line: 800

from sklearn.svm._base import BaseSVC


def test_input(pred_input):
    assert np.array([[2.3], [1.9], [1.8], [0.2], [1.5], [0.2], [2.1], [2.1], [1.8], [1.2], [0.2], [2.3], [0.4], [1.8], [1.3], [1.6], [0.3], [2.1], [1. ], [1.5], [2. ], [1.4], [1.8], [1.4], [2.4], [0.2], [0.2], [0.2], [0.2], [0.4], [1.9], [2.3], [1.3], [1.3], [1.5], [2. ], [0.3], [2.1]])==pred_input['args']['X'], 'Prediction failed!'
    
    
