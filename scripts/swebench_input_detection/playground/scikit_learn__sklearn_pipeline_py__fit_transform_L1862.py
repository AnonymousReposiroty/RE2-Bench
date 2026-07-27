# Problem: scikit_learn__sklearn_pipeline_py__fit_transform_L1862
# Module: sklearn.pipeline
# Function: fit_transform
# Line: 1862

import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator


def parse_array(arr_str):
    """Parse a string representation of numpy array."""
    if arr_str is None:
        return None
    if isinstance(arr_str, str):
        import re
        # Replace standalone 'nan' with 'np.nan'
        cleaned = re.sub(r'(?<![\w.])nan(?![\w])', 'np.nan', arr_str)
        return np.array(eval(cleaned))
    return np.array(arr_str)


def create_feature_union_gt():
    """
    Create the ground truth FeatureUnion with specified parameters.
    """
    transformer_list = [
        ("simpleimputer", SimpleImputer(strategy='most_frequent')),
        ("missingindicator", MissingIndicator())
    ]
    
    fu = FeatureUnion(
        transformer_list=transformer_list,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True
    )
    
    return fu


def parse_transformer(transformer_str):
    """Parse a transformer from its string representation."""
    if 'SimpleImputer' in transformer_str:
        if "strategy='most_frequent'" in transformer_str:
            return SimpleImputer(strategy='most_frequent')
        elif "strategy='mean'" in transformer_str:
            return SimpleImputer(strategy='mean')
        elif "strategy='median'" in transformer_str:
            return SimpleImputer(strategy='median')
        else:
            return SimpleImputer()
    elif 'MissingIndicator' in transformer_str:
        return MissingIndicator()
    else:
        raise ValueError(f"Unknown transformer: {transformer_str}")


def test_input(pred_input):
    # Create ground truth FeatureUnion
    fu_gt = create_feature_union_gt()

    # Ground truth X input
    X_gt = np.array([[np.nan, 1.], [1., np.nan]])
    y_gt = None

    # Call fit_transform on ground truth
    result_gt = fu_gt.fit_transform(X_gt, y_gt)

    # Create predicted FeatureUnion from input
    transformer_list_pred = []
    for name, transformer_str in pred_input['self'].get('transformer_list', []):
        transformer = parse_transformer(transformer_str)
        transformer_list_pred.append((name, transformer))

    fu_pred = FeatureUnion(
        transformer_list=transformer_list_pred,
        n_jobs=pred_input['self'].get('n_jobs', None),
        transformer_weights=pred_input['self'].get('transformer_weights', None),
        verbose=pred_input['self'].get('verbose', False),
        verbose_feature_names_out=pred_input['self'].get('verbose_feature_names_out', True)
    )

    # Parse predicted X and y from input
    X_pred = parse_array(pred_input['args'].get('X'))
    y_pred = pred_input['args'].get('y')
    if y_pred == 'null' or y_pred is None:
        y_pred = None
    else:
        y_pred = parse_array(y_pred)

    # Call fit_transform on predicted
    result_pred = fu_pred.fit_transform(X_pred, y_pred)

    # Compare results
    assert np.allclose(result_gt, result_pred, equal_nan=True), \
        f'Result mismatch!\nGT:\n{result_gt}\nPred:\n{result_pred}'

    print('Test passed!')


if __name__ == '__main__':
    # Test with ground truth input
    gt_input = {
        "self": {
            "transformer_list": [
                ["simpleimputer", "SimpleImputer(strategy='most_frequent')"],
                ["missingindicator", "MissingIndicator()"]
            ],
            "n_jobs": None,
            "transformer_weights": None,
            "verbose": False,
            "verbose_feature_names_out": True
        },
        "args": {
            "X": "[[nan,  1.],[ 1., nan]]",
            "y": None
        },
        "kwargs": {}
    }
    test_input(gt_input)
