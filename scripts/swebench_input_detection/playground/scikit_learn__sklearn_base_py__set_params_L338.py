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

# Problem: scikit-learn@@sklearn_base.py@@set_params_L338
# Module: sklearn.base
# Function: set_params
# Line: 338

from sklearn.base import BaseEstimator


def test_input(pred_input):
    obj_ins = BaseEstimator()
    obj_ins.n_clusters = 3
    obj_ins.svd_method = 'randomized'
    obj_ins.n_svd_vecs = None
    obj_ins.mini_batch = False
    obj_ins.init = 'k-means++'
    obj_ins.n_init = 3
    obj_ins.random_state = 42
    obj_ins.method = 'bistochastic'
    obj_ins.n_components = 6
    obj_ins.n_best = 3
    obj_ins_pred = BaseEstimator()
    obj_ins_pred.n_clusters = pred_input['self']['n_clusters']
    obj_ins_pred.svd_method = pred_input['self']['svd_method']
    obj_ins_pred.n_svd_vecs = pred_input['self']['n_svd_vecs']
    obj_ins_pred.mini_batch = pred_input['self']['mini_batch']
    obj_ins_pred.init = pred_input['self']['init']
    obj_ins_pred.n_init = pred_input['self']['n_init']
    obj_ins_pred.random_state = pred_input['self']['random_state']
    obj_ins_pred.method = pred_input['self']['method']
    obj_ins_pred.n_components = pred_input['self']['n_components']
    obj_ins_pred.n_best = pred_input['self']['n_best']
    # obj_ins_pred = BaseEstimator(n_clusters = pred_input['self']['n_clusters'], svd_method = pred_input['self']['svd_method'], n_svd_vecs = pred_input['self']['n_svd_vecs'], mini_batch = pred_input['self']['mini_batch'], init = pred_input['self']['init'], n_init = pred_input['self']['n_init'], random_state = pred_input['self']['random_state'], method = pred_input['self']['method'], n_components = pred_input['self']['n_components'], n_best = pred_input['self']['n_best'])
    assert obj_ins.set_params(method = 'log')==obj_ins_pred.set_params(method = pred_input['kwargs']['method']), 'Prediction failed!'
 
