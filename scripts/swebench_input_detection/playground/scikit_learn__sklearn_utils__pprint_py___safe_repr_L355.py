import numpy as np
# Problem: scikit-learn@@sklearn_utils__pprint.py@@_safe_repr_L355
# Module: sklearn.utils._pprint
# Function: _safe_repr
# Line: 355

from sklearn.utils._pprint import _safe_repr


def test_input(pred_input):
    assert _safe_repr(object = 'dictionary', context = {'124132985094608': 1}, maxlevels = None, level = 1, changed_only = True)==_safe_repr(object = pred_input['args']['object'], context = pred_input['args']['context'], maxlevels = pred_input['args']['maxlevels'], level = pred_input['args']['level'], changed_only = pred_input['args']['changed_only']), 'Prediction failed!'
