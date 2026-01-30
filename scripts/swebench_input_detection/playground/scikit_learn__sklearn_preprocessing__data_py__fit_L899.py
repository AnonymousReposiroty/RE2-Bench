import numpy as np

# Problem: scikit-learn@@sklearn_preprocessing__data.py@@fit_L899
# Module: sklearn.preprocessing._data
# Function: fit
# Line: 899

from sklearn.preprocessing._data import StandardScaler

nan = np.nan
def test_input(pred_input):
    obj_ins = StandardScaler(with_mean = True, with_std = True, copy = True)
    obj_ins_pred = StandardScaler(with_mean = pred_input['self']['with_mean'], with_std = pred_input['self']['with_std'], copy = pred_input['self']['copy'])
    assert obj_ins.fit(X = np.array([[1, nan], [0, 0]], dtype=object), y = None, sample_weight = None)==obj_ins_pred.fit(X = pred_input['args']['X'], y = pred_input['args']['y'], sample_weight = pred_input['args']['sample_weight']), 'Prediction failed!'
