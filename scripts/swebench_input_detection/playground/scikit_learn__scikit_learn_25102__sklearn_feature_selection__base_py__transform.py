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

# Problem: scikit-learn__scikit-learn-25102@@sklearn.feature_selection._base.py@@transform
# Benchmark: Swebench
# Module: sklearn.feature_selection._base
# Function: transform

from sklearn.feature_selection._base import SelectorMixin


def test_input(pred_input):
    obj_ins = SelectorMixin()
    obj_ins.score_func = {}
    obj_ins.percentile = 67
    obj_ins.n_features_in_ = 3
    obj_ins.scores_ = string2Array('[0 0 1]')
    obj_ins.pvalues_ = string2Array('[0 0 1]')
    obj_ins_pred = SelectorMixin()
    obj_ins_pred.score_func = pred_input['self']['score_func']
    obj_ins_pred.percentile = pred_input['self']['percentile']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred.scores_ = pred_input['self']['scores_']
    obj_ins_pred.pvalues_ = pred_input['self']['pvalues_']
    assert obj_ins.transform(X = [])==obj_ins_pred.transform(X = pred_input['args']['X']), 'Prediction failed!'
    

