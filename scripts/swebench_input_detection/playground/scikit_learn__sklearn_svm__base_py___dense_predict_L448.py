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

# Problem: scikit-learn@@sklearn_svm__base.py@@_dense_predict_L448
# Module: sklearn.svm._base
# Function: _dense_predict
# Line: 448

from sklearn.svm._base import BaseLibSVM


def test_input(pred_input):
    obj_ins = BaseLibSVM(kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, tol = 0.001, C = 1.0, nu = 0.0, epsilon = 0.0, shrinking = True, probability = False, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, random_state = 16930041)
    obj_ins.decision_function_shape = 'ovr'
    obj_ins.break_ties = False
    obj_ins._sparse = False
    obj_ins.n_features_in_ = 1
    obj_ins.class_weight_ = np.array([1., 1., 1.])
    obj_ins.classes_ = np.array([0, 1, 2])
    obj_ins._gamma = 1.7539247982371269
    obj_ins.support_ = np.array([ 4, 21, 54, 3, 6, 12, 16, 25, 38, 39, 55, 56, 65, 66, 5, 8, 10, 14, 20, 26, 28, 44, 52, 68], dtype=np.dtype("int32"))
    obj_ins.support_vectors_ = np.array([[0.6], [0.5], [0.1], [1.8], [1.6], [1.7], [1. ], [1.6], [1. ], [1.5], [1.5], [1. ], [1.5], [1.3], [1.4], [1.8], [1.7], [1.8], [1.5], [1.8], [1.6], [1.5], [1.8], [2.5]])
    obj_ins._n_support = np.array([ 3, 11, 10], dtype=np.dtype("int32"))
    obj_ins.dual_coef_ = np.array([[ 3. , 0.35879615, 0. , -0.01974129, -0. , -0. , -0.3565403 , -0. , -1. , -0. , -0. , -1.98251456, -0. , -0. , -1. , -0. , -0. , -0. , -0. , -0. , -0. , -0.01264989, -0. , -0.67915217], [ 1.23675666, 0. , 0.4550454 , 1. , 3. , 2. , 0. , 2. , 0. , 3. , 1. , 0. , 1. , 0.67783947, -1. , -1.67783947, -1. , -1. , -2. , -3. , -1. , -1. , -2. , -0. ]])
    obj_ins.intercept_ = np.array([-0.15192272, -0.2007334 , -0.42362183])
    obj_ins._probA = np.array([], dtype=np.dtype("float64"))
    obj_ins._probB = np.array([], dtype=np.dtype("float64"))
    obj_ins.fit_status_ = 0
    obj_ins._num_iter = np.array([ 7, 14, 16], dtype=np.dtype("int32"))
    obj_ins.shape_fit_ = [112, 1]
    obj_ins._intercept_ = np.array([-0.15192272, -0.2007334 , -0.42362183])
    obj_ins._dual_coef_ = np.array([[ 3. , 0.35879615, 0. , -0.01974129, -0. , -0. , -0.3565403 , -0. , -1. , -0. , -0. , -1.98251456, -0. , -0. , -1. , -0. , -0. , -0. , -0. , -0. , -0. , -0.01264989, -0. , -0.67915217], [ 1.23675666, 0. , 0.4550454 , 1. , 3. , 2. , 0. , 2. , 0. , 3. , 1. , 0. , 1. , 0.67783947, -1. , -1.67783947, -1. , -1. , -2. , -3. , -1. , -1. , -2. , -0. ]])
    obj_ins.n_iter_ = np.array([ 7, 14, 16], dtype=np.dtype("int32"))
    obj_ins_pred = BaseLibSVM(kernel = pred_input['self']['kernel'], degree = pred_input['self']['degree'], gamma = pred_input['self']['gamma'], coef0 = pred_input['self']['coef0'], tol = pred_input['self']['tol'], C = pred_input['self']['C'], nu = pred_input['self']['nu'], epsilon = pred_input['self']['epsilon'], shrinking = pred_input['self']['shrinking'], probability = pred_input['self']['probability'], cache_size = pred_input['self']['cache_size'], class_weight = pred_input['self']['class_weight'], verbose = pred_input['self']['verbose'], max_iter = pred_input['self']['max_iter'], random_state = pred_input['self']['random_state'])
    obj_ins_pred.decision_function_shape = pred_input['self']['decision_function_shape']
    obj_ins_pred.break_ties = pred_input['self']['break_ties']
    obj_ins_pred._sparse = pred_input['self']['_sparse']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred.class_weight_ = pred_input['self']['class_weight_']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    obj_ins_pred._gamma = pred_input['self']['_gamma']
    obj_ins_pred.support_ = pred_input['self']['support_']
    obj_ins_pred.support_vectors_ = pred_input['self']['support_vectors_']
    obj_ins_pred._n_support = pred_input['self']['_n_support']
    obj_ins_pred.dual_coef_ = pred_input['self']['dual_coef_']
    obj_ins_pred.intercept_ = pred_input['self']['intercept_']
    obj_ins_pred._probA = pred_input['self']['_probA']
    obj_ins_pred._probB = pred_input['self']['_probB']
    obj_ins_pred.fit_status_ = pred_input['self']['fit_status_']
    obj_ins_pred._num_iter = pred_input['self']['_num_iter']
    obj_ins_pred.shape_fit_ = pred_input['self']['shape_fit_']
    obj_ins_pred._intercept_ = pred_input['self']['_intercept_']
    obj_ins_pred._dual_coef_ = pred_input['self']['_dual_coef_']
    obj_ins_pred.n_iter_ = pred_input['self']['n_iter_']
    assert obj_ins._dense_predict(X = np.array([[2.3], [1.9], [1.8], [0.2], [1.5], [0.2], [2.1], [2.1], [1.8], [1.2], [0.2], [2.3], [0.4], [1.8], [1.3], [1.6], [0.3], [2.1], [1. ], [1.5], [2. ], [1.4], [1.8], [1.4], [2.4], [0.2], [0.2], [0.2], [0.2], [0.4], [1.9], [2.3], [1.3], [1.3], [1.5], [2. ], [0.3], [2.1]]))==obj_ins_pred._dense_predict(X = pred_input['args']['X']), 'Prediction failed!'
    
    
    
obj_ins = BaseLibSVM(kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, tol = 0.001, C = 1.0, nu = 0.0, epsilon = 0.0, shrinking = True, probability = False, cache_size = 200, class_weight = None, verbose = False, max_iter = -1, random_state = 16930041)
obj_ins.decision_function_shape = 'ovr'
obj_ins.break_ties = False
obj_ins._sparse = False
obj_ins.n_features_in_ = 1
obj_ins.class_weight_ = np.array([1., 1., 1.])
obj_ins.classes_ = np.array([0, 1, 2])
obj_ins._gamma = 1.7539247982371269
obj_ins.support_ = np.array([ 4, 21, 54, 3, 6, 12, 16, 25, 38, 39, 55, 56, 65, 66, 5, 8, 10, 14, 20, 26, 28, 44, 52, 68], dtype=np.dtype("int32"))
obj_ins.support_vectors_ = np.array([[0.6], [0.5], [0.1], [1.8], [1.6], [1.7], [1. ], [1.6], [1. ], [1.5], [1.5], [1. ], [1.5], [1.3], [1.4], [1.8], [1.7], [1.8], [1.5], [1.8], [1.6], [1.5], [1.8], [2.5]])
obj_ins._n_support = np.array([ 3, 11, 10], dtype=np.dtype("int32"))
obj_ins.dual_coef_ = np.array([[ 3. , 0.35879615, 0. , -0.01974129, -0. , -0. , -0.3565403 , -0. , -1. , -0. , -0. , -1.98251456, -0. , -0. , -1. , -0. , -0. , -0. , -0. , -0. , -0. , -0.01264989, -0. , -0.67915217], [ 1.23675666, 0. , 0.4550454 , 1. , 3. , 2. , 0. , 2. , 0. , 3. , 1. , 0. , 1. , 0.67783947, -1. , -1.67783947, -1. , -1. , -2. , -3. , -1. , -1. , -2. , -0. ]])
obj_ins.intercept_ = np.array([-0.15192272, -0.2007334 , -0.42362183])
obj_ins._probA = np.array([], dtype=np.dtype("float64"))
obj_ins._probB = np.array([], dtype=np.dtype("float64"))
obj_ins.fit_status_ = 0
obj_ins._num_iter = np.array([ 7, 14, 16], dtype=np.dtype("int32"))
obj_ins.shape_fit_ = [112, 1]
obj_ins._intercept_ = np.array([-0.15192272, -0.2007334 , -0.42362183])
obj_ins._dual_coef_ = np.array([[ 3. , 0.35879615, 0. , -0.01974129, -0. , -0. , -0.3565403 , -0. , -1. , -0. , -0. , -1.98251456, -0. , -0. , -1. , -0. , -0. , -0. , -0. , -0. , -0. , -0.01264989, -0. , -0.67915217], [ 1.23675666, 0. , 0.4550454 , 1. , 3. , 2. , 0. , 2. , 0. , 3. , 1. , 0. , 1. , 0.67783947, -1. , -1.67783947, -1. , -1. , -2. , -3. , -1. , -1. , -2. , -0. ]])
obj_ins.n_iter_ = np.array([ 7, 14, 16], dtype=np.dtype("int32"))

obj_ins._dense_predict(X = np.array([[2.3], [1.9], [1.8], [0.2], [1.5], [0.2], [2.1], [2.1], [1.8], [1.2], [0.2], [2.3], [0.4], [1.8], [1.3], [1.6], [0.3], [2.1], [1. ], [1.5], [2. ], [1.4], [1.8], [1.4], [2.4], [0.2], [0.2], [0.2], [0.2], [0.4], [1.9], [2.3], [1.3], [1.3], [1.5], [2. ], [0.3], [2.1]]))