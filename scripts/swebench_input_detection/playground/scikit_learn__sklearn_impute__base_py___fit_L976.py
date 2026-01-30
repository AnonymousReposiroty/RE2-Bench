import numpy as np

# Problem: scikit-learn@@sklearn_impute__base.py@@_fit_L976
# Module: sklearn.impute._base
# Function: _fit
# Line: 976

from sklearn.impute._base import MissingIndicator


def test_input(pred_input):
    obj_ins = MissingIndicator(missing_values = np.nan, features = 'missing-only', sparse = 'auto', error_on_new = False)
    obj_ins_pred = MissingIndicator(missing_values = pred_input['self']['missing_values'], features = pred_input['self']['features'], sparse = pred_input['self']['sparse'], error_on_new = pred_input['self']['error_on_new'])
    assert obj_ins._fit(X = np.array([[ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False]]), y = None, precomputed = True)==obj_ins_pred._fit(X = pred_input['args']['X'], y = pred_input['args']['y'], precomputed = pred_input['args']['precomputed']), 'Prediction failed!'
