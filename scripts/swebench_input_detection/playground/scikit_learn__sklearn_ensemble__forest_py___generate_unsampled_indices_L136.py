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

# Problem: scikit-learn@@sklearn_ensemble__forest.py@@_generate_unsampled_indices_L136
# Module: sklearn.ensemble._forest
# Function: _generate_unsampled_indices
# Line: 136

from sklearn.ensemble._forest import _generate_unsampled_indices


def test_input(pred_input):
    assert _generate_unsampled_indices(random_state = 209652396, n_samples = 150, n_samples_bootstrap = 150)==_generate_unsampled_indices(random_state = pred_input['args']['random_state'], n_samples = pred_input['args']['n_samples'], n_samples_bootstrap = pred_input['args']['n_samples_bootstrap']), 'Prediction failed!'