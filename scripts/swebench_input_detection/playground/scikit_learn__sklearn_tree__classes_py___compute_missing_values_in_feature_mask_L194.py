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

# Problem: scikit-learn@@sklearn_tree__classes.py@@_compute_missing_values_in_feature_mask_L194
# Module: sklearn.tree._classes
# Function: _compute_missing_values_in_feature_mask
# Line: 194

from sklearn.tree._classes import BaseDecisionTree


def test_input(pred_input):
    obj_ins = BaseDecisionTree(criterion = 'gini', splitter = 'best', max_depth = 2, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, max_leaf_nodes = None, random_state = 643999637, min_impurity_decrease = 0.0, class_weight = None, ccp_alpha = 0.0, monotonic_cst = None)
    obj_ins.n_features_in_ = 4
    obj_ins_pred = BaseDecisionTree(criterion = pred_input['self']['criterion'], splitter = pred_input['self']['splitter'], max_depth = pred_input['self']['max_depth'], min_samples_split = pred_input['self']['min_samples_split'], min_samples_leaf = pred_input['self']['min_samples_leaf'], min_weight_fraction_leaf = pred_input['self']['min_weight_fraction_leaf'], max_features = pred_input['self']['max_features'], max_leaf_nodes = pred_input['self']['max_leaf_nodes'], random_state = pred_input['self']['random_state'], min_impurity_decrease = pred_input['self']['min_impurity_decrease'], class_weight = pred_input['self']['class_weight'], ccp_alpha = pred_input['self']['ccp_alpha'], monotonic_cst = pred_input['self']['monotonic_cst'])
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    assert obj_ins._compute_missing_values_in_feature_mask(X = np.array([[3.1, 0.2, 0.2, 4.8], [3.4, 2.3, 2.3, 6.2], [2.9, 1.3, 1.3, 6.4], [4.2, 0.2, 0.2, 5.5], [3.2, 1.8, 1.8, 5.9], [3.5, 0.6, 0.6, 5. ], [2.6, 1.4, 1.4, 6.1], [3.4, 1.6, 1.6, 6. ], [3.8, 2. , 2. , 7.9], [3.7, 0.4, 0.4, 5.1], [3. , 1.8, 1.8, 6. ], [2.8, 1.3, 1.3, 5.7], [3. , 0.1, 0.1, 4.3], [2.4, 1. , 1. , 5.5], [2.5, 1.7, 1.7, 4.9], [3.6, 2.5, 2.5, 7.2], [3.2, 0.2, 0.2, 4.7], [3. , 1.7, 1.7, 6.7], [3.1, 0.2, 0.2, 4.9], [2.9, 1.8, 1.8, 6.3], [2.5, 1.1, 1.1, 5.6], [2.7, 1. , 1. , 5.8], [3. , 2.2, 2.2, 6.5], [3.1, 1.8, 1.8, 6.4], [3.9, 0.4, 0.4, 5.4], [2.5, 2. , 2. , 5.7], [2.9, 1.3, 1.3, 5.6], [3.2, 2. , 2. , 6.5], [2.3, 0.3, 0.3, 4.5], [3.3, 0.2, 0.2, 5. ], [2.2, 1.5, 1.5, 6. ], [2.8, 1.5, 1.5, 6.5], [3.3, 0.5, 0.5, 5.1], [3.1, 2.1, 2.1, 6.9], [2.8, 2. , 2. , 5.6], [3.6, 0.1, 0.1, 4.9], [3.3, 1.6, 1.6, 6.3], [2.9, 1.8, 1.8, 7.3], [3.2, 0.2, 0.2, 4.6], [3. , 1.6, 1.6, 7.2], [3.5, 0.3, 0.3, 5. ], [3.7, 0.2, 0.2, 5.3], [3. , 1.4, 1.4, 6.6], [3.2, 0.2, 0.2, 4.4], [2.8, 1.2, 1.2, 6.1], [2.3, 1.3, 1.3, 5.5], [4. , 0.2, 0.2, 5.8], [3. , 2.1, 2.1, 7.6], [2.7, 1.2, 1.2, 5.8], [3. , 1.8, 1.8, 6.5], [3.2, 1.5, 1.5, 6.4], [3.2, 0.2, 0.2, 5. ], [2.8, 1.9, 1.9, 7.4], [3.2, 0.2, 0.2, 4.7], [3.8, 2.2, 2.2, 7.7], [2.2, 1. , 1. , 6. ], [3.4, 0.4, 0.4, 5.4], [2.9, 1.3, 1.3, 6.6], [3.1, 2.3, 2.3, 6.9], [3.4, 0.2, 0.2, 5. ], [2.6, 1. , 1. , 5.7], [2.6, 1.2, 1.2, 5.5], [3. , 1.5, 1.5, 5.9], [3.3, 2.5, 2.5, 6.3], [2.4, 1.1, 1.1, 5.5], [3.5, 0.2, 0.2, 5.5], [3.5, 0.2, 0.2, 5.1], [2.8, 2.4, 2.4, 5.8], [2.8, 2.2, 2.2, 6.4], [2.3, 1. , 1. , 5. ], [2.8, 1.5, 1.5, 6.3], [3.1, 0.2, 0.2, 4.6], [2.7, 1.4, 1.4, 5.2], [2.9, 0.2, 0.2, 4.4], [3.1, 1.5, 1.5, 6.9], [3. , 1.5, 1.5, 5.4], [4.1, 0.1, 0.1, 5.2], [2.5, 1.9, 1.9, 6.3], [3. , 2.3, 2.3, 6.7], [2.5, 1.3, 1.3, 5.5], [3.5, 0.2, 0.2, 5.2], [3.4, 0.2, 0.2, 5.1], [2.9, 1.4, 1.4, 6.1], [3.8, 0.3, 0.3, 5.1], [3.9, 0.4, 0.4, 5.4], [3.4, 0.4, 0.4, 5. ], [3.2, 2.3, 2.3, 6.4], [2.5, 1.8, 1.8, 6.7], [3.1, 2.4, 2.4, 6.7], [3.5, 0.3, 0.3, 5.1], [3. , 0.1, 0.1, 4.8], [2.2, 1.5, 1.5, 6.2], [2.4, 1. , 1. , 4.9], [3.1, 0.1, 0.1, 4.9], [3.4, 0.2, 0.2, 5.4], [2.5, 1.1, 1.1, 5.1], [3.6, 0.2, 0.2, 4.6], [2.9, 1.3, 1.3, 5.7], [3. , 1.3, 1.3, 5.6], [2.3, 1.3, 1.3, 6.3], [2.6, 1.2, 1.2, 5.8], [3.2, 1.4, 1.4, 7. ], [3.4, 0.2, 0.2, 4.8], [2.7, 1.9, 1.9, 6.4], [3.8, 0.2, 0.2, 5.1], [3.8, 0.3, 0.3, 5.7], [3. , 1.5, 1.5, 5.6], [2.8, 1.3, 1.3, 6.1], [3.2, 2.3, 2.3, 6.8], [3.2, 1.8, 1.8, 7.2], [3.1, 1.4, 1.4, 6.7], [3.3, 2.5, 2.5, 6.7]], dtype=float32), estimator_name = None)==obj_ins_pred._compute_missing_values_in_feature_mask(X = pred_input['args']['X'], estimator_name = pred_input['args']['estimator_name']), 'Prediction failed!'
    