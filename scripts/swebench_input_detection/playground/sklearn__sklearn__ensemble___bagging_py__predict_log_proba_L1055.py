# Problem: sklearn@@sklearn_ensemble__bagging.py@@predict_log_proba_L1055
# Module: sklearn.ensemble._bagging
# Function: BaggingClassifier.predict_log_proba
# Line: 1055

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris


class CustomSVC(SVC):
    """Custom SVC that supports predict_proba by default."""

    def __init__(self, kernel='rbf', random_state=None, **kwargs):
        super().__init__(kernel=kernel, probability=True, random_state=random_state, **kwargs)

    def __repr__(self):
        return f"CustomSVC(kernel='{self.kernel}')"


def test_input(pred_input):
    """
    Test case for BaggingClassifier.predict_log_proba method.

    Ground truth input:
    - BaggingClassifier with n_estimators=10, CustomSVC as base estimator
    - X: 38 samples with 4 features
    - 3 classes

    Expected: returns log-probability matrix of shape (n_samples, n_classes)
    """
    # Load iris data for training
    iris = load_iris()
    X_train = iris.data[:112]  # Use 112 samples as indicated by _n_samples
    y_train = iris.target[:112]

    # Create ground truth BaggingClassifier
    clf_gt = BaggingClassifier(
        estimator=CustomSVC(kernel='linear'),
        n_estimators=10,
        max_samples=1.0,
        max_features=4,
        bootstrap=True,
        bootstrap_features=True,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=1,
        verbose=0
    )
    clf_gt.fit(X_train, y_train)

    # Ground truth X for prediction (38 samples)
    X_gt = np.array(eval(pred_input['args']['X']))

    # Call predict_log_proba with ground truth
    result_gt = clf_gt.predict_log_proba(X_gt)

    # Create predicted BaggingClassifier
    clf_pred = BaggingClassifier(
        estimator=CustomSVC(kernel='linear'),
        n_estimators=pred_input['self']['n_estimators'],
        max_samples=pred_input['self']['max_samples'],
        max_features=pred_input['self']['max_features'],
        bootstrap=pred_input['self']['bootstrap'],
        bootstrap_features=pred_input['self']['bootstrap_features'],
        oob_score=pred_input['self']['oob_score'],
        warm_start=pred_input['self']['warm_start'],
        n_jobs=pred_input['self']['n_jobs'],
        random_state=pred_input['self']['random_state'],
        verbose=pred_input['self']['verbose']
    )
    clf_pred.fit(X_train, y_train)

    # Call predict_log_proba with predicted input
    result_pred = clf_pred.predict_log_proba(X_gt)

    # Compare shapes
    assert result_gt.shape == result_pred.shape, \
        f'Shape mismatch! Expected {result_gt.shape}, got {result_pred.shape}'

    # Both should have shape (n_samples, n_classes) = (38, 3)
    assert result_gt.shape == (38, 3), f'Expected shape (38, 3), got {result_gt.shape}'

    # Values should be close (same random_state should give same results)
    assert np.allclose(result_gt, result_pred, rtol=1e-5), \
        f'Values mismatch!\nExpected:\n{result_gt[:3]}\nGot:\n{result_pred[:3]}'

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for BaggingClassifier.predict_log_proba method output prediction.

    Ground truth:
    - BaggingClassifier with n_estimators=10, CustomSVC as base estimator
    - X: 38 samples with 4 features
    - return: log-probability matrix of shape (38, 3)
    """
    # Load iris data for training
    iris = load_iris()
    X_train = iris.data[:112]
    y_train = iris.target[:112]

    # Create BaggingClassifier
    clf = BaggingClassifier(
        estimator=CustomSVC(kernel='linear'),
        n_estimators=10,
        max_features=4,
        bootstrap=True,
        bootstrap_features=True,
        random_state=1
    )
    clf.fit(X_train, y_train)

    # X for prediction (standardized iris data subset)
    X_test = iris.data[112:150]  # remaining samples
    # Use the first 38 for consistency
    X_test = X_test[:38] if len(X_test) >= 38 else X_test

    # Call predict_log_proba
    result_gt = clf.predict_log_proba(X_test)

    # Compare with prediction
    pred_array = np.array(pred_output) if not isinstance(pred_output, np.ndarray) else pred_output

    assert result_gt.shape == pred_array.shape, \
        f'Shape mismatch! Expected {result_gt.shape}, got {pred_array.shape}'

    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'estimator': "CustomSVC(kernel='linear')",
            'n_estimators': 10,
            'estimator_params': [],
            'max_samples': 1.0,
            'max_features': 4,
            'bootstrap': True,
            'bootstrap_features': True,
            'oob_score': False,
            'warm_start': False,
            'n_jobs': None,
            'random_state': 1,
            'verbose': 0,
            'n_features_in_': 4,
            '_n_samples': 112,
            'classes_': '[0, 1, 2]',
            'n_classes_': 3
        },
        'args': {
            'X': '[[ 2.24968346e+00, -1.05276654e+00,  1.78583195e+00,  1.44883158e+00],[-5.25060772e-02, -8.22569778e-01,  7.62758269e-01,  9.22302838e-01],[ 3.10997534e-01, -1.31979479e-01,  6.49083415e-01,  7.90670654e-01],[-7.79513300e-01,  7.88807586e-01, -1.34022653e+00, -1.31544430e+00],[ 1.03800476e+00,  9.82172869e-02,  5.35408562e-01,  3.95774101e-01],[-1.74885626e+00, -1.31979479e-01, -1.39706395e+00, -1.31544430e+00],[ 1.15917263e+00, -1.31979479e-01,  9.90107977e-01,  1.18556721e+00],[ 6.74501145e-01, -5.92373012e-01,  1.04694540e+00,  1.18556721e+00],[ 6.86617933e-02, -1.31979479e-01,  7.62758269e-01,  7.90670654e-01],[-1.73673948e-01, -1.31979479e-01,  2.51221427e-01,  8.77547895e-04],[-1.02184904e+00, -1.31979479e-01, -1.22655167e+00, -1.31544430e+00],[ 2.24968346e+00, -1.31979479e-01,  1.33113254e+00,  1.44883158e+00],[-1.73673948e-01,  3.09077525e+00, -1.28338910e+00, -1.05217993e+00],[ 5.53333275e-01, -8.22569778e-01,  6.49083415e-01,  7.90670654e-01],[-2.94841818e-01, -8.22569778e-01,  2.51221427e-01,  1.32509732e-01],[ 1.89829664e-01, -8.22569778e-01,  7.62758269e-01,  5.27406285e-01],[-1.50652052e+00,  7.88807586e-01, -1.34022653e+00, -1.18381211e+00],[ 1.03800476e+00,  5.58610819e-01,  1.10378283e+00,  1.18556721e+00],[-1.02184904e+00, -2.43394714e+00, -1.46640561e-01, -2.62386821e-01],[ 1.89829664e-01, -3.62176246e-01,  4.21733708e-01,  3.95774101e-01],[ 2.24968346e+00, -5.92373012e-01,  1.67215710e+00,  1.05393502e+00],[ 3.10997534e-01, -1.31979479e-01,  4.78571135e-01,  2.64141916e-01],[ 4.32165405e-01, -5.92373012e-01,  5.92245988e-01,  7.90670654e-01],[ 1.15917263e+00, -5.92373012e-01,  5.92245988e-01,  2.64141916e-01],[ 5.53333275e-01,  7.88807586e-01,  1.04694540e+00,  1.58046376e+00],[-1.26418478e+00,  7.88807586e-01, -1.05603939e+00, -1.31544430e+00],[-5.37177559e-01,  1.47939788e+00, -1.28338910e+00, -1.31544430e+00],[-1.02184904e+00,  1.24920112e+00, -1.34022653e+00, -1.31544430e+00],[-1.14301691e+00, -1.31979479e-01, -1.34022653e+00, -1.31544430e+00],[-9.00681170e-01,  1.70959465e+00, -1.05603939e+00, -1.05217993e+00],[-5.25060772e-02, -8.22569778e-01,  7.62758269e-01,  9.22302838e-01],[ 1.28034050e+00,  3.28414053e-01,  1.10378283e+00,  1.44883158e+00],[ 4.32165405e-01, -3.62176246e-01,  3.08058854e-01,  1.32509732e-01],[-1.73673948e-01, -5.92373012e-01,  1.94384000e-01,  1.32509732e-01],[ 5.53333275e-01, -1.28296331e+00,  6.49083415e-01,  3.95774101e-01],[ 7.95669016e-01, -1.31979479e-01,  8.19595696e-01,  1.05393502e+00],[-1.26418478e+00, -1.31979479e-01, -1.34022653e+00, -1.18381211e+00],[ 1.52267624e+00, -1.31979479e-01,  1.21745768e+00,  1.18556721e+00]]'
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    iris = load_iris()
    X_train = iris.data[:112]
    y_train = iris.target[:112]

    clf = BaggingClassifier(
        estimator=CustomSVC(kernel='linear'),
        n_estimators=10,
        max_features=4,
        bootstrap=True,
        bootstrap_features=True,
        random_state=1
    )
    clf.fit(X_train, y_train)

    X_test = np.array(eval(gt_input['args']['X']))
    gt_output = clf.predict_log_proba(X_test)

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output shape: {gt_output.shape}")
    print(f"Ground truth output (first 3 rows):\n{gt_output[:3]}")

    test_output(gt_output)
