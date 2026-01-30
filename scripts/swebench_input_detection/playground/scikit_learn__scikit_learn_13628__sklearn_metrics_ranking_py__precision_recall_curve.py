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

# Problem: scikit-learn__scikit-learn-13628@@sklearn.metrics.ranking.py@@precision_recall_curve
# Benchmark: Swebench
# Module: sklearn.metrics.ranking
# Function: precision_recall_curve

from sklearn.metrics import precision_recall_curve

def test_input(pred_input):
    assert precision_recall_curve(y_true = None, probas_pred = None, pos_label = 1, sample_weight = None)==precision_recall_curve(y_true = pred_input['args']['y_true'], probas_pred = pred_input['args']['probas_pred'], pos_label = pred_input['kwargs']['pos_label'], sample_weight = pred_input['kwargs']['sample_weight']), 'Prediction failed!'