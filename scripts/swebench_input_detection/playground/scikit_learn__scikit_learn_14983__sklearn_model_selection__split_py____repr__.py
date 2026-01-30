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

from sklearn.model_selection._split import BaseCrossValidator



def test_input(pred_input):
	obj_ins = BaseCrossValidator(test_fold = string2Array('[1 1 2 2]'), unique_folds = string2Array('[1 2]'))
	obj_ins_pred = BaseCrossValidator(test_fold = pred_input['self']['test_fold'], unique_folds = pred_input['self']['unique_folds'])
	assert obj_ins.__repr__()==obj_ins_pred.__repr__(), 'Prediction failed!'
 
