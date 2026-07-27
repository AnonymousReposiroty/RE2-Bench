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

from sklearn.kernel_approximation import Nystroem


def test_input(pred_input):
	obj_ins = Nystroem(kernel = 'polynomial', gamma = None, coef0 = 0.1, degree = 3.1, kernel_params = None, n_components = 10, random_state = None)
	obj_ins_pred = Nystroem(kernel = pred_input['self']['kernel'], gamma = pred_input['self']['gamma'], coef0 = pred_input['self']['coef0'], degree = pred_input['self']['degree'], kernel_params = pred_input['self']['kernel_params'], n_components = pred_input['self']['n_components'], random_state = pred_input['self']['random_state'])
	assert obj_ins._get_kernel_params()==obj_ins_pred._get_kernel_params(), 'Prediction failed!'

