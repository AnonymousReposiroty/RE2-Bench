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

# Problem: scikit-learn@@sklearn_decomposition__pca.py@@_fit_full_L544
# Module: sklearn.decomposition._pca
# Function: _fit_full
# Line: 544

from sklearn.decomposition._pca import PCA


def test_input(pred_input):
    obj_ins = PCA(n_components = 2, copy = True, whiten = True, svd_solver = 'auto', tol = 0.0, iterated_power = 'auto', n_oversamples = 10, power_iteration_normalizer = 'auto', random_state = 'RandomState(MT19937) at 0x70E5F440CB40')
    obj_ins.n_features_in_ = 2
    obj_ins._fit_svd_solver = 'covariance_eigh'
    obj_ins_pred = PCA(n_components = pred_input['self']['n_components'], copy = pred_input['self']['copy'], whiten = pred_input['self']['whiten'], svd_solver = pred_input['self']['svd_solver'], tol = pred_input['self']['tol'], iterated_power = pred_input['self']['iterated_power'], n_oversamples = pred_input['self']['n_oversamples'], power_iteration_normalizer = pred_input['self']['power_iteration_normalizer'], random_state = pred_input['self']['random_state'])
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._fit_svd_solver = pred_input['self']['_fit_svd_solver']
    assert obj_ins._fit_full(X = np.array([[-0.77421405, -0.44262622],[ 0.82038213,  0.62271584],[ 1.09332522,  0.26233302],[-0.76568212, -0.65764402],[-0.88574246, -0.59004679],[-0.76019433, -0.66180433]]), n_components = 2)==obj_ins_pred._fit_full(X = pred_input['args']['X'], n_components = pred_input['args']['n_components']), 'Prediction failed!'
