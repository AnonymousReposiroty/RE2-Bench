# Problem: scikit_learn__sklearn_impute__knn_py__transform_L254
# Module: sklearn.impute._knn
# Function: transform
# Line: 254

import numpy as np
from sklearn.impute import KNNImputer, MissingIndicator


def parse_array(arr_str):
    """Parse a string representation of numpy array."""
    if isinstance(arr_str, str):
        # Clean up the string and evaluate
        # Use regex to only replace standalone 'nan' not 'np.nan'
        import re
        cleaned = re.sub(r'(?<![\w.])nan(?![\w])', 'np.nan', arr_str)
        return np.array(eval(cleaned))
    return np.array(arr_str)


def create_imputer_gt():
    """
    Create the ground truth KNNImputer with specified parameters and fitted state.
    """
    imputer = KNNImputer(
        missing_values=np.nan,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
        copy=True,
        add_indicator=True,
        keep_empty_features=False
    )

    # Set fitted attributes
    imputer.n_features_in_ = 5
    imputer._fit_X = np.array([
        [np.nan, 1., 5., np.nan, 1.],
        [2., np.nan, 1., np.nan, 2.],
        [6., 3., np.nan, np.nan, 3.],
        [1., 2., 9., np.nan, 4.]
    ])
    imputer._mask_fit_X = np.array([
        [True, False, False, True, False],
        [False, True, False, True, False],
        [False, False, True, True, False],
        [False, False, False, True, False]
    ])
    imputer._valid_mask = np.array([True, True, True, False, True])

    # Set up the MissingIndicator
    imputer.indicator_ = MissingIndicator(error_on_new=False)
    imputer.indicator_._fit(imputer._mask_fit_X, precomputed=True)

    return imputer


def test_input(pred_input):
    # Create ground truth imputer
    imputer_gt = create_imputer_gt()

    # Ground truth X input
    X_gt = np.array([
        [np.nan, 1., 5., np.nan, 1.],
        [2., np.nan, 1., np.nan, 2.],
        [6., 3., np.nan, np.nan, 3.],
        [1., 2., 9., np.nan, 4.]
    ])

    # Call transform on ground truth
    result_gt = imputer_gt.transform(X_gt)

    # Create predicted imputer from input
    imputer_pred = KNNImputer(
        missing_values=np.nan if pred_input['self'].get('missing_values') == 'NaN' or 
                                 (isinstance(pred_input['self'].get('missing_values'), float) and 
                                  np.isnan(pred_input['self'].get('missing_values'))) else pred_input['self'].get('missing_values', np.nan),
        n_neighbors=pred_input['self'].get('n_neighbors', 5),
        weights=pred_input['self'].get('weights', 'uniform'),
        metric=pred_input['self'].get('metric', 'nan_euclidean'),
        copy=pred_input['self'].get('copy', True),
        add_indicator=pred_input['self'].get('add_indicator', False),
        keep_empty_features=pred_input['self'].get('keep_empty_features', False)
    )

    # Set fitted attributes
    imputer_pred.n_features_in_ = pred_input['self'].get('n_features_in_', 5)
    imputer_pred._fit_X = parse_array(pred_input['self'].get('_fit_X'))
    imputer_pred._mask_fit_X = parse_array(pred_input['self'].get('_mask_fit_X'))
    imputer_pred._valid_mask = parse_array(pred_input['self'].get('_valid_mask'))

    # Set up the MissingIndicator if add_indicator is True
    if pred_input['self'].get('add_indicator', False):
        imputer_pred.indicator_ = MissingIndicator(error_on_new=False)
        imputer_pred.indicator_._fit(imputer_pred._mask_fit_X, precomputed=True)

    # Parse predicted X from input
    X_pred = parse_array(pred_input['args']['X'])

    # Call transform on predicted
    result_pred = imputer_pred.transform(X_pred)

    # Compare results
    assert np.allclose(result_gt, result_pred, equal_nan=True), \
        f'Result mismatch!\nGT:\n{result_gt}\nPred:\n{result_pred}'

    print('Test passed!')


if __name__ == '__main__':
    # Test with ground truth input
    gt_input = {
        "self": {
            "missing_values": np.nan,
            "add_indicator": True,
            "keep_empty_features": False,
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "nan_euclidean",
            "copy": True,
            "n_features_in_": 5,
            "_fit_X": "[[nan,  1.,  5., nan,  1.],[ 2., nan,  1., nan,  2.],[ 6.,  3., nan, nan,  3.],[ 1.,  2.,  9., nan,  4.]]",
            "_mask_fit_X": "[[ True, False, False,  True, False],[False,  True, False,  True, False],[False, False,  True,  True, False],[False, False, False,  True, False]]",
            "_valid_mask": "[ True,  True,  True, False,  True]",
            "indicator_": "MissingIndicator(error_on_new=False)"
        },
        "args": {
            "X": "[[nan,  1.,  5., nan,  1.],[ 2., nan,  1., nan,  2.],[ 6.,  3., nan, nan,  3.],[ 1.,  2.,  9., nan,  4.]]"
        },
        "kwargs": {}
    }
    test_input(gt_input)
