# Problem: scikit-learn@@sklearn.preprocessing._encoders.py@@_compute_n_features_outs_L959
# Module: sklearn.preprocessing._encoders
# Function: _compute_n_features_outs
# Line: 959

from sklearn.preprocessing import OneHotEncoder
import numpy as np


def create_encoder():
    """
    Create a OneHotEncoder with the ground truth configuration:
    - categories: 'auto'
    - sparse_output: True
    - dtype: np.float64
    - handle_unknown: 'error'
    - drop: None
    - min_frequency: None
    - max_categories: None
    - feature_name_combiner: 'concat'
    """
    encoder = OneHotEncoder(
        categories='auto',
        sparse_output=True,
        dtype=np.float64,
        handle_unknown='error',
        drop=None,
        min_frequency=None,
        max_categories=None,
        feature_name_combiner='concat',
    )

    # Set internal state as if fit() was called
    encoder._infrequent_enabled = False
    encoder.n_features_in_ = 1
    encoder.categories_ = [np.array(['a', 'b'])]
    encoder._drop_idx_after_grouping = None
    encoder.drop_idx_ = None

    return encoder


def test_input(pred_input):
    # Create ground truth encoder
    encoder_gt = create_encoder()

    # Create predicted encoder with pred_input values
    encoder_pred = OneHotEncoder(
        categories=pred_input['self']['categories'],
        sparse_output=pred_input['self']['sparse_output'],
        dtype=np.float64,  # dtype is passed as string, use np.float64 directly
        handle_unknown=pred_input['self']['handle_unknown'],
        drop=pred_input['self']['drop'],
        min_frequency=pred_input['self']['min_frequency'],
        max_categories=pred_input['self']['max_categories'],
        feature_name_combiner=pred_input['self']['feature_name_combiner'],
    )

    # Set internal state from pred_input
    encoder_pred._infrequent_enabled = pred_input['self']['_infrequent_enabled']
    encoder_pred.n_features_in_ = pred_input['self']['n_features_in_']

    # Parse categories_ from string representation
    categories_str = pred_input['self']['categories_'][0]
    # categories_str is like "['a', 'b']", parse it
    categories_list = eval(categories_str)
    encoder_pred.categories_ = [np.array(categories_list)]

    encoder_pred._drop_idx_after_grouping = pred_input['self']['_drop_idx_after_grouping']
    encoder_pred.drop_idx_ = pred_input['self']['drop_idx_']

    # Call _compute_n_features_outs on both and compare results
    result_gt = encoder_gt._compute_n_features_outs()
    result_pred = encoder_pred._compute_n_features_outs()

    assert result_gt == result_pred, 'Prediction failed!'
