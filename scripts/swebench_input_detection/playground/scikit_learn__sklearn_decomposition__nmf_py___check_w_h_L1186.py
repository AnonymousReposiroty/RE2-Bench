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

# Problem: scikit-learn@@sklearn_decomposition__nmf.py@@_check_w_h_L1186
# Module: sklearn.decomposition._nmf
# Function: _check_w_h
# Line: 1186

from sklearn.decomposition._nmf import _BaseNMF


def test_input(pred_input):
    obj_ins = _BaseNMF(n_components = 'auto', init = None, beta_loss = 'frobenius', tol = 0.0001, max_iter = 1, random_state = None, alpha_W = 0.0, alpha_H = 'same', l1_ratio = 0.0, verbose = 0)
    obj_ins.solver = 'cd'
    obj_ins.shuffle = False
    obj_ins.n_features_in_ = 2
    obj_ins._n_components = 'auto'
    obj_ins._beta_loss = 2
    obj_ins_pred = _BaseNMF(n_components = pred_input['self']['n_components'], init = pred_input['self']['init'], beta_loss = pred_input['self']['beta_loss'], tol = pred_input['self']['tol'], max_iter = pred_input['self']['max_iter'], random_state = pred_input['self']['random_state'], alpha_W = pred_input['self']['alpha_W'], alpha_H = pred_input['self']['alpha_H'], l1_ratio = pred_input['self']['l1_ratio'], verbose = pred_input['self']['verbose'])
    obj_ins_pred.solver = pred_input['self']['solver']
    obj_ins_pred.shuffle = pred_input['self']['shuffle']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_components = pred_input['self']['_n_components']
    obj_ins_pred._beta_loss = pred_input['self']['_beta_loss']
    assert obj_ins._check_w_h(X = np.array([[1., 1.], [1., 1.]]), W = None, H = None, update_H = True)==obj_ins_pred._check_w_h(X = pred_input['args']['X'], W = pred_input['args']['W'], H = pred_input['args']['H'], update_H = pred_input['args']['update_H']), 'Prediction failed!'
