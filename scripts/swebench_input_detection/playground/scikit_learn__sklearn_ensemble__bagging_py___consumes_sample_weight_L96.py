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

# Problem: scikit-learn@@sklearn_ensemble__bagging.py@@_consumes_sample_weight_L96
# Module: sklearn.ensemble._bagging
# Function: _consumes_sample_weight
# Line: 96

from sklearn.utils.metadata_routing import (
    _routing_enabled,
    get_routing_for_object
)
from sklearn.utils.validation import (
    has_fit_parameter
)

def _consumes_sample_weight(estimator):
    if _routing_enabled():
        request_or_router = get_routing_for_object(estimator)
        consumes_sample_weight = request_or_router.consumes("fit", ("sample_weight",))
    else:
        consumes_sample_weight = has_fit_parameter(estimator, "sample_weight")
    return consumes_sample_weight


def test_input(pred_input):
    assert _consumes_sample_weight(estimator = 'KNeighborsClassifier()')==_consumes_sample_weight(estimator = pred_input['args']['estimator']), 'Prediction failed!'
