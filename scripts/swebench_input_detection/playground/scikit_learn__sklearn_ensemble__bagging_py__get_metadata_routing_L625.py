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

# Problem: scikit-learn@@sklearn_ensemble__bagging.py@@get_metadata_routing_L625
# Module: sklearn.ensemble._bagging
# Function: get_metadata_routing
# Line: 625

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class ConsumingClassifierWithoutPredictProba(BaseEstimator, ClassifierMixin):
    def __init__(self, registry=None):
        self.registry = registry

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # Always predicts class 1 for determinism; adjust if you need variety.
        return np.full(X.shape[0], self.classes_[0], dtype=self.classes_.dtype)


def test_input(pred_input):
    obj_ins = BaggingClassifier(
    estimator=ConsumingClassifierWithoutPredictProba(registry=None),
    n_estimators=10,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
)

    obj_ins.estimator_params = []
    obj_ins.n_features_in_ = 2
    obj_ins._n_samples = 3
    obj_ins.classes_ = np.array([1, 2, 3])
    obj_ins.n_classes_ = 3
    obj_ins._max_samples = 3
    obj_ins._max_features = 2
    obj_ins._sample_weight = None
    obj_ins.estimator_ = ConsumingClassifierWithoutPredictProba(registry=None)
    obj_ins.estimator_.classes_ = obj_ins.classes_

    obj_ins.estimators_ = [ConsumingClassifierWithoutPredictProba(registry=None) for _ in range(10)]
    for est in obj_ins.estimators_:
        est.classes_ = obj_ins.classes_
    obj_ins.estimators_features_ = [np.array([0, 1]) for _ in range(10)]

    obj_ins._seeds = np.array([
        670970498, 1954729455, 2126718252, 2110439493, 1878596078,
        2108026088, 1263330277, 352756841, 635828192, 1318191639
    ], dtype=np.int64)
    obj_ins_pred = BaggingClassifier(estimator = pred_input['self']['estimator'], n_estimators = pred_input['self']['n_estimators'], max_samples = pred_input['self']['max_samples'], max_features = pred_input['self']['max_features'], bootstrap = pred_input['self']['bootstrap'], bootstrap_features = pred_input['self']['bootstrap_features'], oob_score = pred_input['self']['oob_score'], warm_start = pred_input['self']['warm_start'], n_jobs = pred_input['self']['n_jobs'], random_state = pred_input['self']['random_state'], verbose = pred_input['self']['verbose'])
    obj_ins_pred.estimator_params = pred_input['self']['estimator_params']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_samples = pred_input['self']['_n_samples']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    obj_ins_pred.n_classes_ = pred_input['self']['n_classes_']
    obj_ins_pred.estimator_ = pred_input['self']['estimator_']
    obj_ins_pred._max_samples = pred_input['self']['_max_samples']
    obj_ins_pred._max_features = pred_input['self']['_max_features']
    obj_ins_pred._sample_weight = pred_input['self']['_sample_weight']
    obj_ins_pred.estimators_ = pred_input['self']['estimators_']
    obj_ins_pred.estimators_features_ = pred_input['self']['estimators_features_']
    obj_ins_pred._seeds = pred_input['self']['_seeds']
    assert obj_ins.get_metadata_routing()==obj_ins_pred.get_metadata_routing(), 'Prediction failed!'
    
