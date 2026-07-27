# Problem: scikit_learn__scikit_learn__sklearn_ensemble__forest_py___validate_y_class_weight_L829
# Module: sklearn.ensemble._forest
# Function: _validate_y_class_weight
# Line: 829

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_classifier():
    """
    Create a RandomForestClassifier with the specified parameters.
    """
    clf = RandomForestClassifier(
        n_estimators=5,
        criterion="gini",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    )
    # Set additional attributes that are set during fit
    clf.n_features_in_ = 20
    clf._n_samples = 10
    clf.n_outputs_ = 1
    return clf


def test_input(pred_input):
    # Ground truth y input
    y_gt = np.array([[0], [0], [1], [0], [1], [0], [1], [0], [1], [1]])

    # Create ground truth classifier and call method
    clf_gt = create_classifier()
    result_y_gt, result_weight_gt = clf_gt._validate_y_class_weight(y_gt)

    # Parse predicted y from input
    y_pred_str = pred_input['args']['y']
    if isinstance(y_pred_str, str):
        y_pred = np.array(eval(y_pred_str))
    else:
        y_pred = np.array(y_pred_str)

    # Create predicted classifier with predicted self attributes
    clf_pred = RandomForestClassifier(
        n_estimators=pred_input['self'].get('n_estimators', 5),
        criterion=pred_input['self'].get('criterion', 'gini'),
        max_depth=pred_input['self'].get('max_depth', 3),
        min_samples_split=pred_input['self'].get('min_samples_split', 2),
        min_samples_leaf=pred_input['self'].get('min_samples_leaf', 1),
        min_weight_fraction_leaf=pred_input['self'].get('min_weight_fraction_leaf', 0.0),
        max_features=pred_input['self'].get('max_features', 'sqrt'),
        max_leaf_nodes=pred_input['self'].get('max_leaf_nodes', None),
        min_impurity_decrease=pred_input['self'].get('min_impurity_decrease', 0.0),
        bootstrap=pred_input['self'].get('bootstrap', True),
        oob_score=pred_input['self'].get('oob_score', False),
        n_jobs=pred_input['self'].get('n_jobs', None),
        random_state=pred_input['self'].get('random_state', None),
        verbose=pred_input['self'].get('verbose', 0),
        warm_start=pred_input['self'].get('warm_start', False),
        class_weight=pred_input['self'].get('class_weight', None),
        max_samples=pred_input['self'].get('max_samples', None),
        ccp_alpha=pred_input['self'].get('ccp_alpha', 0.0),
        monotonic_cst=pred_input['self'].get('monotonic_cst', None),
    )
    clf_pred.n_features_in_ = pred_input['self'].get('n_features_in_', 20)
    clf_pred._n_samples = pred_input['self'].get('_n_samples', 10)
    clf_pred.n_outputs_ = pred_input['self'].get('n_outputs_', 1)

    # Call method with predicted input
    result_y_pred, result_weight_pred = clf_pred._validate_y_class_weight(y_pred)

    # Compare results
    assert np.array_equal(result_y_gt, result_y_pred), f'y result mismatch! GT: {result_y_gt}, Pred: {result_y_pred}'

    # Compare expanded_class_weight (both should be None given class_weight=None)
    if result_weight_gt is None:
        assert result_weight_pred is None, f'Weight mismatch! GT: None, Pred: {result_weight_pred}'
    else:
        assert np.allclose(result_weight_gt, result_weight_pred), f'Weight mismatch! GT: {result_weight_gt}, Pred: {result_weight_pred}'

    # Also verify the classifier attributes were set correctly
    assert all(
        np.array_equal(a, b) for a, b in zip(clf_gt.classes_, clf_pred.classes_)
    ), 'classes_ mismatch!'
    assert clf_gt.n_classes_ == clf_pred.n_classes_, 'n_classes_ mismatch!'

    print('Test passed!')


if __name__ == '__main__':
    # Test with ground truth input
    gt_input = {
        "self": {
            "estimator": "DecisionTreeClassifier()",
            "n_estimators": 5,
            "estimator_params": [
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst"
            ],
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": None,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "max_samples": None,
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "monotonic_cst": None,
            "ccp_alpha": 0.0,
            "n_features_in_": 20,
            "_n_samples": 10,
            "n_outputs_": 1
        },
        "args": {
            "y": "[[0],[0],[1],[0],[1],[0],[1],[0],[1],[1]]"
        },
        "kwargs": {}
    }
    test_input(gt_input)
