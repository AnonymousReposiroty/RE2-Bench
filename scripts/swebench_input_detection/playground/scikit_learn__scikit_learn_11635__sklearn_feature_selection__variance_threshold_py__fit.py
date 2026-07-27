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

from sklearn.feature_selection import VarianceThreshold

def test_input(pred_input):
	obj_ins = VarianceThreshold(threshold = 0.4)
	obj_ins_pred = VarianceThreshold(threshold = pred_input['self']['threshold'])
	assert obj_ins.fit(
    X={
        '_shape': (3, 5),
        'maxprint': 50,
        'data':    np.array([1, 2, 3, 4, 2, 2, 3, 5, 1, 1, 2, 4]),
        'indices': np.array([1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 2, 3]),
        'indptr':  np.array([0, 4, 8, 12]),
    }) == obj_ins_pred.fit(X=pred_input['args']['X']), 'Prediction failed!'
