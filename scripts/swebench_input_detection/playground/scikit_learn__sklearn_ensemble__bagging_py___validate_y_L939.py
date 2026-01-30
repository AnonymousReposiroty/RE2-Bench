import numpy as np
from io import StringIO
def string2Array(s):
    s = s.strip()

    # String case (elements quoted, no commas, arbitrary newlines)
    if "'" in s or '"' in s:
        # Remove brackets & quotes, collapse all whitespace to single spaces,
        # then parse as a single whitespace-delimited row of strings.
        payload = (
            s.replace('[', ' ').replace(']', ' ')
             .replace("'", ' ').replace('"', ' ')
        )
        payload = ' '.join(payload.split())
        arr = np.loadtxt(StringIO(payload), dtype=str, comments=None)
        return np.atleast_1d(arr)

    # Numeric case (works for 1D or 2D pretty-printed arrays)
    txt = s.replace('[', '').replace(']', '')
    try:
        return np.loadtxt(StringIO(txt), dtype=float, comments=None)
    except ValueError:
        # If wrapping caused uneven rows, flatten newlines to a single row
        txt_one_line = ' '.join(txt.split())
        return np.loadtxt(StringIO(txt_one_line), dtype=float, comments=None)

# Problem: scikit-learn@@sklearn_ensemble__bagging.py@@_validate_y_L939
# Module: sklearn.ensemble._bagging
# Function: _validate_y
# Line: 939

from sklearn.ensemble._bagging import BaggingClassifier


def test_input(pred_input):
    obj_ins = BaggingClassifier(estimator = None, n_estimators = 2, max_samples = 0.5, max_features = 1, bootstrap = True, bootstrap_features = True, oob_score = False, warm_start = False, n_jobs = None, random_state = 'RandomState(MT19937) at 0x70E5E8725E40', verbose = 0)
    obj_ins.estimator_params = []
    obj_ins.n_features_in_ = 4
    obj_ins._n_samples = 112
    obj_ins_pred = BaggingClassifier(estimator = pred_input['self']['estimator'], n_estimators = pred_input['self']['n_estimators'], max_samples = pred_input['self']['max_samples'], max_features = pred_input['self']['max_features'], bootstrap = pred_input['self']['bootstrap'], bootstrap_features = pred_input['self']['bootstrap_features'], oob_score = pred_input['self']['oob_score'], warm_start = pred_input['self']['warm_start'], n_jobs = pred_input['self']['n_jobs'], random_state = pred_input['self']['random_state'], verbose = pred_input['self']['verbose'])
    obj_ins_pred.estimator_params = pred_input['self']['estimator_params']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_samples = pred_input['self']['_n_samples']
    assert obj_ins._validate_y(y = np.array([0, 2, 1, 0, 1, 0, 2, 1, 2, 0, 2, 1, 0, 1, 2, 2, 0, 1, 0, 2, 1, 1, 2, 2, 0, 2, 1, 2, 0, 0, 2, 1, 0, 2, 2, 0, 1, 2, 0, 2, 0, 0, 1, 0, 1, 1, 0, 2, 1, 2, 1, 0, 2, 0, 2, 1, 0, 1, 2, 0, 1, 1, 1, 2, 1, 0, 0, 2, 2, 1, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 0, 1, 0, 0, 0, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 1, 2, 2, 1, 2]))==obj_ins_pred._validate_y(y = pred_input['args']['y']), 'Prediction failed!'

