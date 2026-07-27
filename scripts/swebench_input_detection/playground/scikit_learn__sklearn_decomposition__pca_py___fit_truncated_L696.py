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

# Problem: scikit-learn@@sklearn_decomposition__pca.py@@_fit_truncated_L696
# Module: sklearn.decomposition._pca
# Function: _fit_truncated
# Line: 696

from sklearn.decomposition._pca import PCA


def test_input(pred_input):
    obj_ins = PCA(n_components = 3, copy = True, whiten = False, svd_solver = 'arpack', tol = 0.0, iterated_power = 'auto', n_oversamples = 10, power_iteration_normalizer = 'auto', random_state = None)
    obj_ins.n_features_in_ = 10
    obj_ins._fit_svd_solver = 'arpack'
    obj_ins_pred = PCA(n_components = pred_input['self']['n_components'], copy = pred_input['self']['copy'], whiten = pred_input['self']['whiten'], svd_solver = pred_input['self']['svd_solver'], tol = pred_input['self']['tol'], iterated_power = pred_input['self']['iterated_power'], n_oversamples = pred_input['self']['n_oversamples'], power_iteration_normalizer = pred_input['self']['power_iteration_normalizer'], random_state = pred_input['self']['random_state'])
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._fit_svd_solver = pred_input['self']['_fit_svd_solver']
    assert obj_ins._fit_truncated(X = np.array([[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258],[0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497,0.18340451, 0.30424224, 0.52475643, 0.43194502, 0.29122914],[0.61185289, 0.13949386, 0.29214465, 0.36636184, 0.45606998,0.78517596, 0.19967378, 0.51423444, 0.59241457, 0.04645041],[0.60754485, 0.17052412, 0.06505159, 0.94888554, 0.96563203,0.80839735, 0.30461377, 0.09767211, 0.68423303, 0.44015249],[0.12203823, 0.49517691, 0.03438852, 0.9093204 , 0.25877998,0.66252228, 0.31171108, 0.52006802, 0.54671028, 0.18485446]]), n_components = 3, svd_solver = 'arpack')==obj_ins_pred._fit_truncated(X = pred_input['args']['X'], n_components = pred_input['args']['n_components'], svd_solver=pred_input['self']['_fit_svd_solver']), 'Prediction failed!'