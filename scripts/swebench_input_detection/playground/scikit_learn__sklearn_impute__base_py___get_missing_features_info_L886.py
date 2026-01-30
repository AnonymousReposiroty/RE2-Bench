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

# Problem: scikit-learn@@sklearn_impute__base.py@@_get_missing_features_info_L886
# Module: sklearn.impute._base
# Function: _get_missing_features_info
# Line: 886

from sklearn.impute._base import MissingIndicator


def test_input(pred_input):
    obj_ins = MissingIndicator(missing_values = np.nan, features = 'missing-only', sparse = 'auto', error_on_new = False)
    obj_ins._precomputed = True
    obj_ins.n_features_in_ = 2
    obj_ins._n_features = 2
    obj_ins_pred = MissingIndicator(missing_values = pred_input['self']['missing_values'], features = pred_input['self']['features'], sparse = pred_input['self']['sparse'], error_on_new = pred_input['self']['error_on_new'])
    obj_ins_pred._precomputed = pred_input['self']['_precomputed']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_features = pred_input['self']['_n_features']
    assert obj_ins._get_missing_features_info(X = np.array([[ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False], [ True, True], [False, False]]))==obj_ins_pred._get_missing_features_info(X = pred_input['args']['X']), 'Prediction failed!'