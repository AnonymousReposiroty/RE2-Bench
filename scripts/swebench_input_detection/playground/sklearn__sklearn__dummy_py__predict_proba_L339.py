# Problem: sklearn@@sklearn_dummy.py@@predict_proba_L339
# Module: sklearn.dummy
# Function: DummyClassifier.predict_proba
# Line: 339

import numpy as np
from sklearn.dummy import DummyClassifier


def test_input(pred_input):
    """
    Test case for DummyClassifier.predict_proba method.

    Ground truth input:
    - DummyClassifier with strategy="prior"
    - classes_: [0, 1, 2]
    - class_prior_: [0.33928571, 0.33035714, 0.33035714]
    - X: 38 samples with 1 feature

    Expected: returns probability matrix of shape (n_samples, n_classes)
    For "prior" strategy, each row equals class_prior_
    """
    # Create training data that produces the desired class_prior_
    # class_prior_ = [0.33928571, 0.33035714, 0.33035714] suggests roughly equal distribution
    # With 3 classes and these priors, we need ~112 samples total (38+37+37)
    np.random.seed(42)
    n_train = 112
    y_train = np.array([0]*38 + [1]*37 + [2]*37)  # This gives approximately the class_prior_
    X_train = np.random.randn(n_train, 1)

    # Create and fit ground truth DummyClassifier
    clf_gt = DummyClassifier(strategy='prior', random_state=967609597)
    clf_gt.fit(X_train, y_train)

    # Ground truth X for prediction
    X_gt = np.array([
        [2.6],[2.7],[3.],[3.4],[3.1],[3.],[3.],[2.8],[3.],[3.],
        [3.],[3.],[4.4],[2.7],[2.7],[2.7],[3.4],[3.3],[2.],[2.9],
        [2.8],[3.],[2.8],[2.8],[3.4],[3.4],[3.7],[3.6],[3.],[3.8],
        [2.7],[3.2],[2.9],[2.8],[2.5],[3.],[3.],[3.]
    ])

    # Call predict_proba with ground truth
    result_gt = clf_gt.predict_proba(X_gt)

    # Create predicted DummyClassifier
    clf_pred = DummyClassifier(
        strategy=pred_input['self']['strategy'],
        random_state=pred_input['self']['random_state']
    )
    clf_pred.fit(X_train, y_train)

    # Parse X from prediction
    X_pred = np.array(eval(pred_input['args']['X']))

    # Call predict_proba with predicted input
    result_pred = clf_pred.predict_proba(X_pred)

    # Compare shapes
    assert result_gt.shape == result_pred.shape, \
        f'Shape mismatch! Expected {result_gt.shape}, got {result_pred.shape}'

    # Both should return same probabilities for "prior" strategy
    assert np.allclose(result_gt, result_pred), \
        f'Values mismatch!\nExpected:\n{result_gt[:3]}\nGot:\n{result_pred[:3]}'

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for DummyClassifier.predict_proba method output prediction.

    Ground truth:
    - DummyClassifier with strategy="prior"
    - class_prior_: [0.33928571, 0.33035714, 0.33035714]
    - X: 38 samples
    - return: probability matrix where each row equals class_prior_
    """
    # For "prior" strategy, output should be class_prior_ repeated for each sample
    n_samples = 38
    class_prior = np.array([0.33928571, 0.33035714, 0.33035714])

    # Expected output: each row is class_prior_
    expected = np.tile(class_prior, (n_samples, 1))

    # Compare with prediction
    pred_array = np.array(pred_output) if not isinstance(pred_output, np.ndarray) else pred_output

    assert expected.shape == pred_array.shape, \
        f'Shape mismatch! Expected {expected.shape}, got {pred_array.shape}'

    assert np.allclose(expected, pred_array, rtol=1e-5), \
        f'Values mismatch!\nExpected:\n{expected[:3]}\nGot:\n{pred_array[:3]}'

    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'strategy': 'prior',
            'random_state': 967609597,
            'constant': None,
            'n_features_in_': 1,
            '_strategy': 'prior',
            'sparse_output_': False,
            'n_outputs_': 1,
            'classes_': [0, 1, 2],
            'n_classes_': 3,
            'class_prior_': [0.33928571, 0.33035714, 0.33035714]
        },
        'args': {
            'X': '[[2.6],[2.7],[3. ],[3.4],[3.1],[3. ],[3. ],[2.8],[3. ],[3. ],[3. ],[3. ],[4.4],[2.7],[2.7],[2.7],[3.4],[3.3],[2. ],[2.9],[2.8],[3. ],[2.8],[2.8],[3.4],[3.4],[3.7],[3.6],[3. ],[3.8],[2.7],[3.2],[2.9],[2.8],[2.5],[3. ],[3. ],[3. ]]'
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test - simulate the expected output for "prior" strategy
    n_samples = 38
    class_prior = np.array([0.33928571, 0.33035714, 0.33035714])
    gt_output = np.tile(class_prior, (n_samples, 1))

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output shape: {gt_output.shape}")
    print(f"Ground truth output (first 3 rows):\n{gt_output[:3]}")

    test_output(gt_output)
