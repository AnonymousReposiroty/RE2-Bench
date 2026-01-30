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

# Problem: scikit-learn@@sklearn_dummy.py@@predict_proba_L339
# Module: sklearn.dummy
# Function: predict_proba
# Line: 339

from sklearn.dummy import DummyClassifier


def test_input(pred_input):
    obj_ins = DummyClassifier(strategy = 'prior', random_state = 967609597, constant = None)
    obj_ins.n_features_in_ = 1
    obj_ins._strategy = 'prior'
    obj_ins.sparse_output_ = False
    obj_ins.n_outputs_ = 1
    obj_ins.classes_ = np.array([0, 1, 2])
    obj_ins.n_classes_ = 3
    obj_ins.class_prior_ = np.array([0.33928571, 0.33035714, 0.33035714])
    obj_ins_pred = DummyClassifier(strategy = pred_input['self']['strategy'], random_state = pred_input['self']['random_state'], constant = pred_input['self']['constant'])
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._strategy = pred_input['self']['_strategy']
    obj_ins_pred.sparse_output_ = pred_input['self']['sparse_output_']
    obj_ins_pred.n_outputs_ = pred_input['self']['n_outputs_']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    obj_ins_pred.n_classes_ = pred_input['self']['n_classes_']
    obj_ins_pred.class_prior_ = pred_input['self']['class_prior_']
    assert obj_ins.predict_proba(X = np.array([[2.6], [2.7], [3. ], [3.4], [3.1], [3. ], [3. ], [2.8], [3. ], [3. ], [3. ], [3. ], [4.4], [2.7], [2.7], [2.7], [3.4], [3.3], [2. ], [2.9], [2.8], [3. ], [2.8], [2.8], [3.4], [3.4], [3.7], [3.6], [3. ], [3.8], [2.7], [3.2], [2.9], [2.8], [2.5], [3. ], [3. ], [3. ]]))==obj_ins_pred.predict_proba(X = pred_input['args']['X']), 'Prediction failed!'