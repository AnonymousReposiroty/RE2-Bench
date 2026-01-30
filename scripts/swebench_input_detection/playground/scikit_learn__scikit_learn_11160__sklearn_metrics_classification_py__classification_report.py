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
	assert classification_report(y_true = string2Array("['red' 'red' 'blue' 'red' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'blue' 'greengreengreengreengreen' 'red'\n 'greengreengreengreengreen' 'red' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'blue' 'red' 'red'\n 'greengreengreengreengreen' 'blue' 'red' 'greengreengreengreengreen'\n 'red' 'red' 'blue' 'greengreengreengreengreen' 'blue' 'red'\n 'greengreengreengreengreen' 'blue' 'greengreengreengreengreen' 'blue'\n 'greengreengreengreengreen' 'greengreengreengreengreen' 'blue' 'blue'\n 'blue' 'blue' 'red' 'blue' 'greengreengreengreengreen' 'red' 'blue'\n 'greengreengreengreengreen' 'blue' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'blue' 'blue' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'red' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'blue' 'blue' 'red' 'blue'\n 'greengreengreengreengreen' 'blue' 'red' 'red' 'blue'\n 'greengreengreengreengreen' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'greengreengreengreengreen' 'blue' 'red'\n 'blue' 'greengreengreengreengreen' 'red' 'red']"), y_pred = string2Array("['red' 'red' 'greengreengreengreengreen' 'red' 'red' 'red' 'blue'\n 'greengreengreengreengreen' 'red' 'red' 'red' 'red' 'blue' 'red' 'red'\n 'blue' 'red' 'red' 'red' 'red' 'greengreengreengreengreen' 'red' 'red'\n 'red' 'blue' 'red' 'blue' 'red' 'greengreengreengreengreen'\n 'greengreengreengreengreen' 'red' 'blue' 'red'\n 'greengreengreengreengreen' 'blue' 'blue' 'blue' 'blue' 'red' 'blue'\n 'red' 'greengreengreengreengreen' 'blue' 'red' 'blue' 'blue' 'blue'\n 'blue' 'greengreengreengreengreen' 'red' 'red' 'red' 'blue' 'red' 'red'\n 'red' 'blue' 'blue' 'red' 'greengreengreengreengreen' 'red' 'blue' 'red'\n 'red' 'blue' 'red' 'red' 'red' 'red' 'blue' 'red' 'blue' 'red' 'red'\n 'red']"))==classification_report(y_true = pred_input['args']['y_true'], y_pred = pred_input['args']['y_pred']), 'Prediction failed!'
