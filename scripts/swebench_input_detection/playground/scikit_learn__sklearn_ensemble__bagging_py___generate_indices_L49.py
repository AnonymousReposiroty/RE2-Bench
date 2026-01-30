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

# Problem: scikit-learn@@sklearn_ensemble__bagging.py@@_generate_indices_L49
# Module: sklearn.ensemble._bagging
# Function: _generate_indices
# Line: 49

from sklearn.ensemble._bagging import _generate_indices


def test_input(pred_input):
    assert _generate_indices(random_state = 'RandomState(MT19937) at 0x70E5E8726140', bootstrap = True, n_population = 4, n_samples = 1)==_generate_indices(random_state = pred_input['args']['random_state'], bootstrap = pred_input['args']['bootstrap'], n_population = pred_input['args']['n_population'], n_samples = pred_input['args']['n_samples']), 'Prediction failed!'