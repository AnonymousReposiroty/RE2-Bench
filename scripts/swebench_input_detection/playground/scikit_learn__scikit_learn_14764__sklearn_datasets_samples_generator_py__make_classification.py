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

from sklearn.datasets import make_classification

def test_input(pred_input):
	assert make_classification(class_sep = 1000000.0, n_redundant = 0, n_repeated = 0, flip_y = 0, shift = 0, scale = 1, shuffle = False, n_samples = 200, n_classes = 4, weights = None, n_features = 2, n_informative = 2, n_clusters_per_class = 1, hypercube = False, random_state = 0)==make_classification(class_sep = pred_input['kwargs']['class_sep'], n_redundant = pred_input['kwargs']['n_redundant'], n_repeated = pred_input['kwargs']['n_repeated'], flip_y = pred_input['kwargs']['flip_y'], shift = pred_input['kwargs']['shift'], scale = pred_input['kwargs']['scale'], shuffle = pred_input['kwargs']['shuffle'], n_samples = pred_input['kwargs']['n_samples'], n_classes = pred_input['kwargs']['n_classes'], weights = pred_input['kwargs']['weights'], n_features = pred_input['kwargs']['n_features'], n_informative = pred_input['kwargs']['n_informative'], n_clusters_per_class = pred_input['kwargs']['n_clusters_per_class'], hypercube = pred_input['kwargs']['hypercube'], random_state = pred_input['kwargs']['random_state']), 'Prediction failed!'
