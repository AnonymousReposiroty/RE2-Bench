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

# Problem: scikit-learn@@sklearn_metrics_cluster__bicluster.py@@_jaccard_L23
# Module: sklearn.metrics.cluster._bicluster
# Function: _jaccard
# Line: 23

from sklearn.metrics.cluster._bicluster import _jaccard


def test_input(pred_input):
    assert _jaccard(a_rows = np.array([False, True, False, True, False, False, False, False, True, False, False, True, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False]), a_cols = np.array([False, False, False, False, True, True, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, True]), b_rows = np.array([False, True, False, True, False, False, False, False, True, False, False, True, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False]), b_cols = np.array([False, False, False, False, True, True, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, True]))==_jaccard(a_rows = pred_input['args']['a_rows'], a_cols = pred_input['args']['a_cols'], b_rows = pred_input['args']['b_rows'], b_cols = pred_input['args']['b_cols']), 'Prediction failed!'
    
