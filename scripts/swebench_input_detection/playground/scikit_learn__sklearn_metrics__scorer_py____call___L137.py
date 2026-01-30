import numpy as np

# Problem: scikit-learn@@sklearn_metrics__scorer.py@@__call___L137
# Module: sklearn.metrics._scorer
# Function: __call__
# Line: 137

from sklearn.metrics._scorer import _MultimetricScorer


def test_input(pred_input):
    obj_ins = _MultimetricScorer()
    obj_ins._scorers = {'score': 'EmpiricalCovariance.score'}
    obj_ins._raise_exc = False
    obj_ins_pred = _MultimetricScorer()
    obj_ins_pred._scorers = pred_input['self']['_scorers']
    obj_ins_pred._raise_exc = pred_input['self']['_raise_exc']
    assert np.array([[ 0.88895051, -0.94884286, -0.77838201, -0.99349011, -0.74907652]])== pred_input['args']['args'], 'Prediction failed!'
    