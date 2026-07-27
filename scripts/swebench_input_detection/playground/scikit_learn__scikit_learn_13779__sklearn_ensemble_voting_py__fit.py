from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier


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


# soft voting needs predict_proba-capable base estimators
estimators = [
    ("d1", DummyClassifier(strategy="prior", random_state=0)),
    ("d2", DummyClassifier(strategy="prior", random_state=1)),
]



def test_input(pred_input):
    obj_ins = VotingClassifier(
    estimators=estimators,
    voting="soft",
    weights=None,
    n_jobs=None,
    flatten_transform=True
    )
    obj_ins_pred = obj_ins = VotingClassifier(
    estimators=estimators,
    voting="soft",
    weights=None,
    n_jobs=None,
    flatten_transform=True
    )
    X = string2Array("[[5.4 3.7 1.5 0.2]\n [4.8 3.4 1.6 0.2]\n [4.8 3.  1.4 0.1]\n [4.3 3.  1.1 0.1]\n [5.8 4.  1.2 0.2]\n [5.7 4.4 1.5 0.4]\n [5.4 3.9 1.3 0.4]\n [5.1 3.5 1.4 0.3]\n [5.7 3.8 1.7 0.3]\n [5.1 3.8 1.5 0.3]\n [5.4 3.4 1.7 0.2]\n [5.1 3.7 1.5 0.4]\n [4.6 3.6 1.  0.2]\n [5.1 3.3 1.7 0.5]\n [4.8 3.4 1.9 0.2]\n [5.  3.  1.6 0.2]\n [5.  3.4 1.6 0.4]\n [5.2 3.5 1.5 0.2]\n [5.2 3.4 1.4 0.2]\n [4.7 3.2 1.6 0.2]\n [4.8 3.1 1.6 0.2]\n [5.4 3.4 1.5 0.4]\n [5.2 4.1 1.5 0.1]\n [5.5 4.2 1.4 0.2]\n [4.9 3.1 1.5 0.2]\n [5.  3.2 1.2 0.2]\n [5.5 3.5 1.3 0.2]\n [4.9 3.6 1.4 0.1]\n [4.4 3.  1.3 0.2]\n [5.1 3.4 1.5 0.2]\n [5.  3.5 1.3 0.3]\n [4.5 2.3 1.3 0.3]\n [4.4 3.2 1.3 0.2]\n [5.  3.5 1.6 0.6]\n [5.1 3.8 1.9 0.4]\n [4.8 3.  1.4 0.3]\n [5.1 3.8 1.6 0.2]\n [4.6 3.2 1.4 0.2]\n [5.3 3.7 1.5 0.2]\n [5.  3.3 1.4 0.2]\n [5.  2.  3.5 1. ]\n [5.9 3.  4.2 1.5]\n [6.  2.2 4.  1. ]\n [6.1 2.9 4.7 1.4]\n [5.6 2.9 3.6 1.3]\n [6.7 3.1 4.4 1.4]\n [5.6 3.  4.5 1.5]\n [5.8 2.7 4.1 1. ]\n [6.2 2.2 4.5 1.5]\n [5.6 2.5 3.9 1.1]\n [5.9 3.2 4.8 1.8]\n [6.1 2.8 4.  1.3]\n [6.3 2.5 4.9 1.5]\n [6.1 2.8 4.7 1.2]\n [6.4 2.9 4.3 1.3]\n [6.6 3.  4.4 1.4]\n [6.8 2.8 4.8 1.4]\n [6.7 3.  5.  1.7]\n [6.  2.9 4.5 1.5]\n [5.7 2.6 3.5 1. ]\n [5.5 2.4 3.8 1.1]\n [5.5 2.4 3.7 1. ]\n [5.8 2.7 3.9 1.2]\n [6.  2.7 5.1 1.6]\n [5.4 3.  4.5 1.5]\n [6.  3.4 4.5 1.6]\n [6.7 3.1 4.7 1.5]\n [6.3 2.3 4.4 1.3]\n [5.6 3.  4.1 1.3]\n [5.5 2.5 4.  1.3]\n [5.5 2.6 4.4 1.2]\n [6.1 3.  4.6 1.4]\n [5.8 2.6 4.  1.2]\n [5.  2.3 3.3 1. ]\n [5.6 2.7 4.2 1.3]\n [5.7 3.  4.2 1.2]\n [5.7 2.9 4.2 1.3]\n [6.2 2.9 4.3 1.3]\n [5.1 2.5 3.  1.1]\n [5.7 2.8 4.1 1.3]\n [6.5 3.2 5.1 2. ]\n [6.4 2.7 5.3 1.9]\n [6.8 3.  5.5 2.1]\n [5.7 2.5 5.  2. ]\n [5.8 2.8 5.1 2.4]\n [6.4 3.2 5.3 2.3]\n [6.5 3.  5.5 1.8]\n [7.7 3.8 6.7 2.2]\n [7.7 2.6 6.9 2.3]\n [6.  2.2 5.  1.5]\n [6.9 3.2 5.7 2.3]\n [5.6 2.8 4.9 2. ]\n [7.7 2.8 6.7 2. ]\n [6.3 2.7 4.9 1.8]\n [6.7 3.3 5.7 2.1]\n [7.2 3.2 6.  1.8]\n [6.2 2.8 4.8 1.8]\n [6.1 3.  4.9 1.8]\n [6.4 2.8 5.6 2.1]\n [7.2 3.  5.8 1.6]\n [7.4 2.8 6.1 1.9]\n [7.9 3.8 6.4 2. ]\n [6.4 2.8 5.6 2.2]\n [6.3 2.8 5.1 1.5]\n [6.1 2.6 5.6 1.4]\n [7.7 3.  6.1 2.3]\n [6.3 3.4 5.6 2.4]\n [6.4 3.1 5.5 1.8]\n [6.  3.  4.8 1.8]\n [6.9 3.1 5.4 2.1]\n [6.7 3.1 5.6 2.4]\n [6.9 3.1 5.1 2.3]\n [5.8 2.7 5.1 1.9]\n [6.8 3.2 5.9 2.3]\n [6.7 3.3 5.7 2.5]\n [6.7 3.  5.2 2.3]\n [6.3 2.5 5.  1.9]\n [6.5 3.  5.2 2. ]\n [6.2 3.4 5.4 2.3]\n [5.9 3.  5.1 1.8]]")
    y=string2Array("[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n 2 2 2 2 2 2 2 2 2]",)
    assert obj_ins.fit(X, y)==obj_ins_pred.fit(pred_input['args']['X'], pred_input['args']['y']), 'Prediction failed!'