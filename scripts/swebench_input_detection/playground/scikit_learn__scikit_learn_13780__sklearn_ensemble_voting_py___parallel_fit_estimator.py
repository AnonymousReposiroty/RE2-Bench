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

# Problem: scikit-learn__scikit-learn-13780@@sklearn.ensemble.voting.py@@_parallel_fit_estimator
# Benchmark: Swebench
# Module: sklearn.ensemble.voting
# Function: _parallel_fit_estimator

from sklearn.ensemble._voting import _parallel_fit_estimator



def test_input(pred_input):
    assert _parallel_fit_estimator(estimator = {'priors': None, 'var_smoothing': 1e-09}, X = string2Array('[[3.5 1.4]\n [3.  1.4]\n [3.2 1.3]\n [3.1 1.5]\n [3.6 1.4]\n [3.9 1.7]\n [3.4 1.4]\n [3.4 1.5]\n [2.9 1.4]\n [3.1 1.5]\n [3.7 1.5]\n [3.4 1.6]\n [3.  1.4]\n [3.  1.1]\n [4.  1.2]\n [4.4 1.5]\n [3.9 1.3]\n [3.5 1.4]\n [3.8 1.7]\n [3.8 1.5]\n [3.1 1.6]\n [3.4 1.5]\n [4.1 1.5]\n [4.2 1.4]\n [3.1 1.5]\n [3.2 1.2]\n [3.5 1.3]\n [3.6 1.4]\n [3.  1.3]\n [3.4 1.5]\n [3.5 1.3]\n [2.3 1.3]\n [3.2 1.3]\n [3.5 1.6]\n [3.8 1.9]\n [3.  1.4]\n [3.8 1.6]\n [3.2 1.4]\n [3.7 1.5]\n [3.3 1.4]\n [3.2 4.7]\n [3.2 4.5]\n [3.1 4.9]\n [2.3 4. ]\n [2.8 4.6]\n [2.8 4.5]\n [3.3 4.7]\n [2.4 3.3]\n [2.9 4.6]\n [2.7 3.9]\n [2.  3.5]\n [3.  4.2]\n [2.2 4. ]\n [2.9 4.7]\n [2.9 3.6]\n [3.1 4.4]\n [3.  4.5]\n [2.7 4.1]\n [2.2 4.5]\n [2.5 3.9]\n [2.4 3.8]\n [2.4 3.7]\n [2.7 3.9]\n [2.7 5.1]\n [3.  4.5]\n [3.4 4.5]\n [3.1 4.7]\n [2.3 4.4]\n [3.  4.1]\n [2.5 4. ]\n [2.6 4.4]\n [3.  4.6]\n [2.6 4. ]\n [2.3 3.3]\n [2.7 4.2]\n [3.  4.2]\n [2.9 4.2]\n [2.9 4.3]\n [2.5 3. ]\n [2.8 4.1]\n [3.3 6. ]\n [2.7 5.1]\n [3.  5.9]\n [2.9 5.6]\n [3.  5.8]\n [3.  6.6]\n [2.5 4.5]\n [2.9 6.3]\n [2.5 5.8]\n [3.6 6.1]\n [3.2 5.1]\n [2.7 5.3]\n [3.  5.5]\n [2.5 5. ]\n [2.8 5.1]\n [3.2 5.3]\n [3.  5.5]\n [3.8 6.7]\n [2.6 6.9]\n [2.2 5. ]\n [2.8 6.1]\n [3.8 6.4]\n [2.8 5.6]\n [2.8 5.1]\n [2.6 5.6]\n [3.  6.1]\n [3.4 5.6]\n [3.1 5.5]\n [3.  4.8]\n [3.1 5.4]\n [3.1 5.6]\n [3.1 5.1]\n [2.7 5.1]\n [3.2 5.9]\n [3.3 5.7]\n [3.  5.2]\n [2.5 5. ]\n [3.  5.2]\n [3.4 5.4]\n [3.  5.1]]'), y = string2Array('[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n 2 2 2 2 2 2 2 2 2]'), sample_weight = None)==_parallel_fit_estimator(estimator = pred_input['args']['estimator'], X = pred_input['args']['X'], y = pred_input['args']['y'], sample_weight = pred_input['kwargs']['sample_weight']), 'Prediction failed!'
