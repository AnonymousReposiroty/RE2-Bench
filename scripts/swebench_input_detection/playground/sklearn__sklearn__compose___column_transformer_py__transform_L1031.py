# Problem: sklearn@@sklearn_compose__column_transformer.py@@transform_L1031
# Module: sklearn.compose._column_transformer
# Function: ColumnTransformer.transform
# Line: 1031

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
    Test case for ColumnTransformer.transform method.

    Ground truth input:
    - ColumnTransformer with one transformer "trans" (Trans()) on columns [0, 1]
    - remainder="drop"
    - X = [[0, 2], [1, 4], [2, 6]]

    Expected: returns transformed array
    """
    # Create ground truth ColumnTransformer
    ct_gt = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop',
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True
    )

    # Ground truth X
    X_gt = np.array([[0, 2], [1, 4], [2, 6]])

    # Fit first (transform requires fitted estimator)
    ct_gt.fit(X_gt)

    # Call transform with ground truth input
    result_gt = ct_gt.transform(X_gt)

    # Create predicted ColumnTransformer
    # Parse columns from input
    columns = pred_input['self']['transformers'][0][2]
    ct_pred = ColumnTransformer(
        transformers=[('trans', Trans(), columns)],
        remainder=pred_input['self']['remainder'],
        sparse_threshold=pred_input['self']['sparse_threshold'],
        n_jobs=pred_input['self']['n_jobs'],
        transformer_weights=pred_input['self']['transformer_weights'],
        verbose=pred_input['self']['verbose'],
        verbose_feature_names_out=pred_input['self']['verbose_feature_names_out']
    )

    # Parse X from prediction
    X_pred = np.array(eval(pred_input['args']['X']))

    # Fit first
    ct_pred.fit(X_pred)

    # Call transform with predicted input
    result_pred = ct_pred.transform(X_pred)

    # Compare results
    assert np.array_equal(result_gt, result_pred), \
        f'Transform results mismatch!\nExpected:\n{result_gt}\nGot:\n{result_pred}'

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for ColumnTransformer.transform method output prediction.

    Ground truth:
    - ColumnTransformer with transformer on columns [0, 1]
    - X = [[0, 2], [1, 4], [2, 6]]
    - return: transformed array (same as input since Trans passes through)
    """
    # Create ground truth ColumnTransformer
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop'
    )

    X = np.array([[0, 2], [1, 4], [2, 6]])

    # Fit first
    ct.fit(X)

    # Call transform
    result_gt = ct.transform(X)

    # Compare with prediction
    pred_array = np.array(pred_output) if not isinstance(pred_output, np.ndarray) else pred_output
    assert np.array_equal(result_gt, pred_array), \
        f'Output mismatch!\nExpected:\n{result_gt}\nGot:\n{pred_array}'

    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'transformers': [['trans', 'Trans()', [0, 1]]],
            'remainder': 'drop',
            'sparse_threshold': 0.3,
            'n_jobs': None,
            'transformer_weights': None,
            'verbose': False,
            'verbose_feature_names_out': True,
            'force_int_remainder_cols': 'deprecated',
            'n_features_in_': 2,
            '_columns': [[0, 1]],
            '_transformer_to_input_indices': {
                'trans': [0, 1],
                'remainder': []
            },
            '_remainder': ['remainder', 'drop', []],
            'sparse_output_': False,
            'transformers_': [['trans', 'Trans()', [0, 1]]],
            'output_indices_': {
                'trans': 'slice(0, 2, None)',
                'remainder': 'slice(0, 0, None)'
            }
        },
        'args': {
            'X': '[[0, 2],[1, 4],[2, 6]]'
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop'
    )
    X = np.array([[0, 2], [1, 4], [2, 6]])
    ct.fit(X)
    gt_output = ct.transform(X)

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output shape: {gt_output.shape}")
    print(f"Ground truth output:\n{gt_output}")

    test_output(gt_output)
