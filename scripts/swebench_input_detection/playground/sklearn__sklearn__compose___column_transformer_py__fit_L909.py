# Problem: sklearn@@sklearn_compose__column_transformer.py@@fit_L909
# Module: sklearn.compose._column_transformer
# Function: ColumnTransformer.fit
# Line: 909

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class Trans(BaseEstimator, TransformerMixin):
    """Simple transformer that passes through data unchanged."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure 2D output
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def __repr__(self):
        return "Trans()"


def test_input(pred_input):
    """
    Test case for ColumnTransformer.fit method.

    Ground truth input:
    - ColumnTransformer with one transformer "trans" (Trans()) on column 0
    - remainder="drop"
    - X = [[0, 2], [1, 4], [2, 6]]
    - y = None

    Expected: returns self (fitted ColumnTransformer)
    """
    # Create ground truth ColumnTransformer
    ct_gt = ColumnTransformer(
        transformers=[('trans', Trans(), 0)],
        remainder='drop',
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True
    )

    # Ground truth X
    X_gt = np.array([[0, 2], [1, 4], [2, 6]])

    # Call fit with ground truth input
    result_gt = ct_gt.fit(X_gt, y=None)

    # Create predicted ColumnTransformer
    ct_pred = ColumnTransformer(
        transformers=[('trans', Trans(), 0)],
        remainder=pred_input['self']['remainder'],
        sparse_threshold=pred_input['self']['sparse_threshold'],
        n_jobs=pred_input['self']['n_jobs'],
        transformer_weights=pred_input['self']['transformer_weights'],
        verbose=pred_input['self']['verbose'],
        verbose_feature_names_out=pred_input['self']['verbose_feature_names_out']
    )

    # Parse X from prediction
    X_pred = np.array(eval(pred_input['args']['X']))

    # Call fit with predicted input
    result_pred = ct_pred.fit(X_pred, y=pred_input['args']['y'])

    # Both should return self
    assert result_gt is ct_gt, "Ground truth fit should return self"
    assert result_pred is ct_pred, "Predicted fit should return self"

    # Check that both are fitted (have necessary attributes)
    assert hasattr(result_gt, 'transformers_'), "Ground truth should be fitted"
    assert hasattr(result_pred, 'transformers_'), "Predicted should be fitted"

    # Check n_features_in_ matches
    assert result_gt.n_features_in_ == result_pred.n_features_in_, \
        f"n_features_in_ mismatch: {result_gt.n_features_in_} vs {result_pred.n_features_in_}"

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for ColumnTransformer.fit method output prediction.

    Ground truth:
    - ColumnTransformer with transformer on column 0
    - X = [[0, 2], [1, 4], [2, 6]]
    - y = None
    - return: self (fitted ColumnTransformer)
    """
    # Create ground truth ColumnTransformer
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), 0)],
        remainder='drop'
    )

    X = np.array([[0, 2], [1, 4], [2, 6]])

    # Call fit
    result_gt = ct.fit(X, y=None)

    # Check that result is self
    assert result_gt is ct, "fit should return self"

    # Verify the predicted output is also a fitted ColumnTransformer
    assert type(result_gt).__name__ == type(pred_output).__name__, \
        f"Type mismatch: {type(result_gt).__name__} vs {type(pred_output).__name__}"

    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'transformers': [['trans', 'Trans()', 0]],
            'remainder': 'drop',
            'sparse_threshold': 0.3,
            'n_jobs': None,
            'transformer_weights': None,
            'verbose': False,
            'verbose_feature_names_out': True,
            'force_int_remainder_cols': 'deprecated',
            'n_features_in_': 2,
            '_columns': [0],
            '_transformer_to_input_indices': {
                'trans': [0],
                'remainder': [1]
            },
            '_remainder': ['remainder', 'drop', [1]],
            'sparse_output_': False,
            'transformers_': [
                ['trans', 'Trans()', 0],
                ['remainder', 'drop', [1]]
            ],
            'output_indices_': {
                'trans': 'slice(0, 1, None)',
                'remainder': 'slice(0, 0, None)'
            }
        },
        'args': {
            'X': '[[0, 2], [1, 4], [2, 6]]',
            'y': None
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), 0)],
        remainder='drop'
    )
    X = np.array([[0, 2], [1, 4], [2, 6]])
    gt_output = ct.fit(X, y=None)

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth n_features_in_: {gt_output.n_features_in_}")
    print(f"Ground truth transformers_: {gt_output.transformers_}")

    test_output(gt_output)
