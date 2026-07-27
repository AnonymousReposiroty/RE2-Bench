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

from sklearn.metrics import classification_report

def test_input(pred_input):
	assert classification_report(y_true = string2Array("['red' 'red' 'blue' 'red' 'green' 'green' 'blue' 'green' 'red' 'green'\n 'red' 'green' 'green' 'green' 'green' 'blue' 'red' 'red' 'green' 'blue'\n 'red' 'green' 'red' 'red' 'blue' 'green' 'blue' 'red' 'green' 'blue'\n 'green' 'blue' 'green' 'green' 'blue' 'blue' 'blue' 'blue' 'red' 'blue'\n 'green' 'red' 'blue' 'green' 'blue' 'green' 'green' 'blue' 'blue' 'green'\n 'green' 'green' 'green' 'red' 'green' 'green' 'blue' 'blue' 'red' 'blue'\n 'green' 'blue' 'red' 'red' 'blue' 'green' 'green' 'green' 'green' 'blue'\n 'red' 'blue' 'green' 'red' 'red']"), y_pred = string2Array("['red' 'red' 'green' 'red' 'red' 'red' 'blue' 'green' 'red' 'red' 'red'\n 'red' 'blue' 'red' 'red' 'blue' 'red' 'red' 'red' 'red' 'green' 'red'\n 'red' 'red' 'blue' 'red' 'blue' 'red' 'green' 'green' 'red' 'blue' 'red'\n 'green' 'blue' 'blue' 'blue' 'blue' 'red' 'blue' 'red' 'green' 'blue'\n 'red' 'blue' 'blue' 'blue' 'blue' 'green' 'red' 'red' 'red' 'blue' 'red'\n 'red' 'red' 'blue' 'blue' 'red' 'green' 'red' 'blue' 'red' 'red' 'blue'\n 'red' 'red' 'red' 'red' 'blue' 'red' 'blue' 'red' 'red' 'red']"))==classification_report(y_true = pred_input['args']['y_true'], y_pred = pred_input['args']['y_pred']), 'Prediction failed!'
