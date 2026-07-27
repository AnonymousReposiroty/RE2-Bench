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

# Problem: scikit-learn__scikit-learn-13143@@sklearn.metrics.classification.py@@f1_score
# Benchmark: Swebench
# Module: sklearn.metrics.classification
# Function: f1_score

from sklearn.metrics import f1_score



def test_input(pred_input):
    assert f1_score(y_true = string2Array('[2 2 0 2 1 1 0 1 2 1 2 1 1 1 1 0 2 2 1 0 2 1 2 2 0 1 0 2 1 0 1 0 1 1 0 0 0\n 0 2 0 1 2 0 1 0 1 1 0 0 1 1 1 1 2 1 1 0 0 2 0 1 0 2 2 0 1 1 1 1 0 2 0 1 2\n 2]'), y_pred = string2Array('[2 2 1 2 2 2 0 1 2 2 2 2 0 2 2 0 2 2 2 2 1 2 2 2 0 2 0 2 1 1 2 0 2 1 0 0 0\n 0 2 0 2 1 0 2 0 0 0 0 1 2 2 2 0 2 2 2 0 0 2 1 2 0 2 2 0 2 2 2 2 0 2 0 2 2\n 2]'), average = 'micro')==f1_score(y_true = pred_input['args']['y_true'], y_pred = pred_input['args']['y_pred'], average = pred_input['kwargs']['average']), 'Prediction failed!'
    
