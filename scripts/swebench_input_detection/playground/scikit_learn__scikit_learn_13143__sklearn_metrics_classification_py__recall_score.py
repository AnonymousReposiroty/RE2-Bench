import numpy as np

# Problem: scikit-learn__scikit-learn-13143@@sklearn.metrics.classification.py@@recall_score
# Benchmark: Swebench
# Module: sklearn.metrics.classification
# Function: recall_score

from sklearn.metrics import recall_score


def test_input(pred_input):
    assert recall_score(y_true = [], y_pred = [], labels = [], average = None)==recall_score(y_true = pred_input['args']['y_true'], y_pred = pred_input['args']['y_pred'], labels = pred_input['kwargs']['labels'], average = pred_input['kwargs']['average']), 'Prediction failed!'
    