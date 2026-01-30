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

from sklearn.metrics.pairwise import euclidean_distances

def test_input(pred_input):
	assert euclidean_distances(X = np.array([[0.56804456, 0.92559664, 0.07103606, 0.0871293 ]]), Y =np.array([[0.96366276, 0.38344152, 0.79172504, 0.52889492]]), squared = True)==euclidean_distances(X = pred_input['args']['X'], Y = pred_input['args']['Y'], squared = pred_input['kwargs']['squared']), 'Prediction failed!'

