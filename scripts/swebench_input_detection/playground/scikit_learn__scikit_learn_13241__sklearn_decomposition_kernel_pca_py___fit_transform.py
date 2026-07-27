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

from sklearn.decomposition.kernel_pca import KernelPCA

def test_input(pred_input):
	obj_ins = KernelPCA(n_components = 4, kernel = 'poly', kernel_params = None, gamma = None, degree = 3, coef0 = 1, alpha = 1.0, fit_inverse_transform = True, eigen_solver = 'arpack', remove_zero_eig = False, tol = 0, max_iter = None, random_state = None, n_jobs = None, copy_X = True)
	obj_ins_pred = KernelPCA(n_components = pred_input['self']['n_components'], kernel = pred_input['self']['kernel'], kernel_params = pred_input['self']['kernel_params'], gamma = pred_input['self']['gamma'], degree = pred_input['self']['degree'], coef0 = pred_input['self']['coef0'], alpha = pred_input['self']['alpha'], fit_inverse_transform = pred_input['self']['fit_inverse_transform'], eigen_solver = pred_input['self']['eigen_solver'], remove_zero_eig = pred_input['self']['remove_zero_eig'], tol = pred_input['self']['tol'], max_iter = pred_input['self']['max_iter'], random_state = pred_input['self']['random_state'], n_jobs = pred_input['self']['n_jobs'], copy_X = pred_input['self']['copy_X'])
	assert obj_ins._fit_transform(K = string2Array('[[2.56139035 2.52117244 2.69797422 2.02911389 2.67068704]\n [2.52117244 2.71967647 2.56295262 1.89198858 2.83722686]\n [2.69797422 2.56295262 3.34510468 1.95856619 2.48090204]\n [2.02911389 1.89198858 1.95856619 2.1869279  1.8532024 ]\n [2.67068704 2.83722686 2.48090204 1.8532024  3.47058014]]'))==obj_ins_pred._fit_transform(K = pred_input['args']['K']), 'Prediction failed!'
 