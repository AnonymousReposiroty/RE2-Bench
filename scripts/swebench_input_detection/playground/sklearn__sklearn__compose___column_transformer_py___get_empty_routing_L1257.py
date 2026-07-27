# Problem: sklearn@@sklearn_compose__column_transformer.py@@_get_empty_routing_L1257
# Module: sklearn.compose._column_transformer
# Function: ColumnTransformer._get_empty_routing
# Line: 1257

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Bunch


class Trans(BaseEstimator, TransformerMixin):
    """Simple transformer that passes through data unchanged."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def __repr__(self):
        return "Trans()"


def test_input(pred_input):
    """
    Test case for ColumnTransformer._get_empty_routing method.

    The method takes no arguments (only self) and returns a Bunch object
    with empty routing for each transformer.

    Ground truth input:
    - ColumnTransformer with transformers defined
    - No additional args (method takes only self)

    Expected: returns Bunch with empty routing dictionaries
    """
    # Create ground truth ColumnTransformer
    ct_gt = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop'
    )

    # Fit first (required for _get_empty_routing to work - needs _columns attribute)
    X_dummy = np.array([[0, 1], [2, 3]])
    ct_gt.fit(X_dummy)

    # Call _get_empty_routing with ground truth
    result_gt = ct_gt._get_empty_routing()

    # Create predicted ColumnTransformer based on input
    transformers = []
    for t in pred_input['self'].get('transformers', []):
        name = t[0]
        columns = t[2]
        transformers.append((name, Trans(), columns))

    ct_pred = ColumnTransformer(
        transformers=transformers if transformers else [('trans', Trans(), [0, 1])],
        remainder=pred_input['self'].get('remainder', 'drop')
    )

    # Fit first (required for _get_empty_routing to work)
    ct_pred.fit(X_dummy)

    # Call _get_empty_routing with predicted input
    result_pred = ct_pred._get_empty_routing()

    # Both should return Bunch objects
    assert isinstance(result_gt, Bunch), f"Expected Bunch, got {type(result_gt)}"
    assert isinstance(result_pred, Bunch), f"Expected Bunch, got {type(result_pred)}"

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for ColumnTransformer._get_empty_routing method output prediction.

    Ground truth:
    - ColumnTransformer with one transformer
    - return: Bunch with empty routing for each transformer
    """
    # Create ground truth ColumnTransformer
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop'
    )

    # Fit first (required for _get_empty_routing)
    X_dummy = np.array([[0, 1], [2, 3]])
    ct.fit(X_dummy)

    # Call _get_empty_routing
    result_gt = ct._get_empty_routing()

    # Check type
    assert isinstance(result_gt, Bunch), f"Expected Bunch, got {type(result_gt)}"
    assert isinstance(pred_output, Bunch), f"Expected Bunch for prediction, got {type(pred_output)}"

    # Check structure matches
    assert set(result_gt.keys()) == set(pred_output.keys()), \
        f"Keys mismatch: {set(result_gt.keys())} vs {set(pred_output.keys())}"

    print("Output test passed!")


if __name__ == "__main__":
    # Test with a proper input for ColumnTransformer
    # Note: _get_empty_routing takes no args, so we only need self info
    gt_input = {
        'self': {
            'transformers': [['trans', 'Trans()', [0, 1]]],
            'remainder': 'drop',
            'sparse_threshold': 0.3,
            'n_jobs': None,
            'transformer_weights': None,
            'verbose': False,
            'verbose_feature_names_out': True
        },
        'args': {},  # _get_empty_routing takes no args
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    ct = ColumnTransformer(
        transformers=[('trans', Trans(), [0, 1])],
        remainder='drop'
    )
    X_dummy = np.array([[0, 1], [2, 3]])
    ct.fit(X_dummy)
    gt_output = ct._get_empty_routing()

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output keys: {list(gt_output.keys())}")
    print(f"Ground truth output: {gt_output}")

    test_output(gt_output)
