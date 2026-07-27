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

from sklearn.model_selection._split import StratifiedKFold


def test_input(pred_input):
	obj_ins = StratifiedKFold(n_splits = 5, shuffle = False, random_state = None)
	obj_ins_pred = StratifiedKFold(n_splits = pred_input['self']['n_splits'], shuffle = pred_input['self']['shuffle'], random_state = pred_input['self']['random_state'])
	assert obj_ins._make_test_folds(X = {'_is_copy': None, '_mgr': {}, '_item_cache': {}, '_attrs': {}}, y = string2Array('[2 1 1 0 0 0 2 1 1 0 2 2 2 2 0 1 1 2 1 0 0 0 0 2 1 1 0 0 2 0 1 1 1 2 0 0 1\n 2 0 0 2 0 1 0 1 1 2 1 2 1 1 1 1 2 2 2 0 2 0 2 1 0 1 0 1 1 2 2 0 0 0 0 2 0\n 1 0 0 2 1 1 2 0 1 1 2 1 0 2 2 1 2 0 0 1 2 0 1 2 2 0 2 0 2 1 0 2 0 1 1 2 2\n 2 0 2 0 0 1 0 2 2 1 2 0 0 0 0 1 1 2 0 0 0 1 2 1 1 2 2 2 1 2 2 1 2 2 1 1 0\n 1 1]'))==obj_ins_pred._make_test_folds(X = pred_input['args']['X'], y = pred_input['args']['y']), 'Prediction failed!'
 
