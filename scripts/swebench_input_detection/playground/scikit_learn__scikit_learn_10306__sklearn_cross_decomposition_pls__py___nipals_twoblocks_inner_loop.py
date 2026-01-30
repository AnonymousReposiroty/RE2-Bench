import numpy as np
from io import StringIO
def string2Array(s):
    return np.loadtxt(StringIO(s.replace('[','').replace(']','')))
import warnings
import numpy as np
from scipy.linalg import pinv2
from sklearn.exceptions import ConvergenceWarning
__all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']

def _nipals_twoblocks_inner_loop(X, Y, mode='A', max_iter=500, tol=1e-06, norm_y_weights=False):
    y_score = Y[:, [0]]
    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps
    while True:
        if mode == 'B':
            if X_pinv is None:
                X_pinv = pinv2(X, check_finite=False)
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        x_score = np.dot(X, x_weights)
        if mode == 'B':
            if Y_pinv is None:
                Y_pinv = pinv2(Y, check_finite=False)
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached', ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return (x_weights, y_weights, ite)

def test_input(pred_input):
	assert _nipals_twoblocks_inner_loop(X = string2Array('[[-0.3400563   0.10832985  0.37518837]\n [ 0.29390487 -0.03694377 -0.34970456]\n [ 0.13679135 -0.1529668  -0.10183731]\n [-0.09063993  0.08158072  0.07635349]]'), Y = string2Array('[[ -2.22044605e-16  -2.49032012e-02]\n [  3.33066907e-16   5.86106290e-02]\n [ -3.33066907e-16  -6.15542361e-02]\n [  4.44089210e-16   2.78468082e-02]]'), mode = 'B', max_iter = 500, tol = 1e-06, norm_y_weights = True)==_nipals_twoblocks_inner_loop(X = pred_input['kwargs']['X'], Y = pred_input['kwargs']['Y'], mode = pred_input['kwargs']['mode'], max_iter = pred_input['kwargs']['max_iter'], tol = pred_input['kwargs']['tol'], norm_y_weights = pred_input['kwargs']['norm_y_weights']), 'Prediction failed!'