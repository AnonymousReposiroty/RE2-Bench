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

# Problem: scikit-learn@@sklearn_utils_multiclass.py@@type_of_target_L240
# Module: sklearn.utils.multiclass
# Function: type_of_target
# Line: 240

from sklearn.utils.multiclass import type_of_target


def test_input(pred_input):
    assert type_of_target(y = np.array([0, 0, 1, 1]), input_name = '', raise_unknown = False)==type_of_target(y = pred_input['args']['y'], input_name = pred_input['args']['input_name'], raise_unknown = pred_input['args']['raise_unknown']), 'Prediction failed!'
