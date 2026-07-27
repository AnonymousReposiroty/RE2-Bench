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

# Problem: scikit-learn@@sklearn_base.py@@get_params_L240
# Module: sklearn.base
# Function: get_params
# Line: 240

from sklearn.base import BaseEstimator


def test_input(pred_input):
    obj_ins = BaseEstimator()
    obj_ins.damping = 0.5
    obj_ins.max_iter = 200
    obj_ins.convergence_iter = 15
    obj_ins.copy = False
    obj_ins.verbose = False
    obj_ins.preference = -36.150962919368084
    obj_ins.affinity = 'precomputed'
    obj_ins.random_state = 0
    obj_ins_pred = BaseEstimator()
    obj_ins_pred.damping = pred_input['self']['damping']
    obj_ins_pred.max_iter = pred_input['self']['max_iter']
    obj_ins_pred.convergence_iter = pred_input['self']['convergence_iter']
    obj_ins_pred.copy = pred_input['self']['copy']
    obj_ins_pred.verbose = pred_input['self']['verbose']
    obj_ins_pred.preference = pred_input['self']['preference']
    obj_ins_pred.affinity = pred_input['self']['affinity']
    obj_ins_pred.random_state = pred_input['self']['random_state']
    assert obj_ins.get_params(deep = False)==obj_ins_pred.get_params(deep = pred_input['args']['deep']), 'Prediction failed!'
