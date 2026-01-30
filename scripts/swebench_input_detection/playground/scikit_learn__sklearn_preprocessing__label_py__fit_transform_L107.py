import numpy as np

# Problem: scikit-learn@@sklearn_preprocessing__label.py@@fit_transform_L107
# Module: sklearn.preprocessing._label
# Function: fit_transform
# Line: 107

from sklearn.preprocessing._label import LabelEncoder


def test_input(pred_input):
    obj_ins = LabelEncoder()
    obj_ins_pred = LabelEncoder()
    assert obj_ins.fit_transform(y = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))==obj_ins_pred.fit_transform(y = pred_input['args']['y']), 'Prediction failed!'
