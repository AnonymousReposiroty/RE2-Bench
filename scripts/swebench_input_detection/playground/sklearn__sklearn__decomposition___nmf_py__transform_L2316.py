# Problem: sklearn@@sklearn_decomposition__nmf.py@@transform_L2316
# Module: sklearn.decomposition._nmf
# Function: MiniBatchNMF.transform
# Line: 2316

import numpy as np
from sklearn.decomposition import MiniBatchNMF


def test_input(pred_input):
    """
    Test case for MiniBatchNMF.transform method.

    Ground truth input:
    - Fitted MiniBatchNMF with n_components=3
    - X: 6x5 matrix of non-negative values

    Expected: returns W matrix of shape (n_samples, n_components)
    """
    # Create ground truth data
    X_gt = np.array([
        [0.49671415, 0.1382643 , 0.64768854, 1.52302986, 0.23415337],
        [0.23413696, 1.57921282, 0.76743473, 0.46947439, 0.54256004],
        [0.46341769, 0.46572975, 0.24196227, 1.91328024, 1.72491783],
        [0.56228753, 1.01283112, 0.31424733, 0.90802408, 1.4123037 ],
        [1.46564877, 0.2257763 , 0.0675282 , 1.42474819, 0.54438272],
        [0.11092259, 1.15099358, 0.37569802, 0.60063869, 0.29169375]
    ])

    # Create and fit ground truth MiniBatchNMF
    nmf_gt = MiniBatchNMF(
        n_components=3,
        init=None,
        beta_loss='frobenius',
        tol=0.001,
        max_iter=200,
        random_state=0,
        verbose=0
    )
    nmf_gt.fit(X_gt)

    # Call transform with ground truth
    result_gt = nmf_gt.transform(X_gt)

    # Create predicted data
    X_pred = np.array(eval(pred_input['args']['X']))

    # Create and fit predicted MiniBatchNMF
    nmf_pred = MiniBatchNMF(
        n_components=pred_input['self']['n_components'],
        init=pred_input['self']['init'],
        beta_loss=pred_input['self']['beta_loss'],
        tol=pred_input['self']['tol'],
        max_iter=pred_input['self']['max_iter'],
        random_state=pred_input['self']['random_state'],
        verbose=pred_input['self']['verbose']
    )
    nmf_pred.fit(X_pred)

    # Call transform with predicted input
    result_pred = nmf_pred.transform(X_pred)

    # Compare shapes
    assert result_gt.shape == result_pred.shape, \
        f'Shape mismatch! Expected {result_gt.shape}, got {result_pred.shape}'

    # Both should have shape (n_samples, n_components)
    assert result_gt.shape == (6, 3), f'Expected shape (6, 3), got {result_gt.shape}'
    assert result_pred.shape == (6, 3), f'Expected shape (6, 3), got {result_pred.shape}'

    print("Input test passed!")


def test_output(pred_output):
    """
    Test case for MiniBatchNMF.transform method output prediction.

    Ground truth:
    - Fitted MiniBatchNMF with n_components=3
    - X: 6x5 matrix
    - return: W matrix of shape (6, 3)
    """
    # Create ground truth data
    X = np.array([
        [0.49671415, 0.1382643 , 0.64768854, 1.52302986, 0.23415337],
        [0.23413696, 1.57921282, 0.76743473, 0.46947439, 0.54256004],
        [0.46341769, 0.46572975, 0.24196227, 1.91328024, 1.72491783],
        [0.56228753, 1.01283112, 0.31424733, 0.90802408, 1.4123037 ],
        [1.46564877, 0.2257763 , 0.0675282 , 1.42474819, 0.54438272],
        [0.11092259, 1.15099358, 0.37569802, 0.60063869, 0.29169375]
    ])

    # Create and fit MiniBatchNMF
    nmf = MiniBatchNMF(n_components=3, random_state=0)
    nmf.fit(X)

    # Call transform
    result_gt = nmf.transform(X)

    # Compare with prediction
    pred_array = np.array(pred_output) if not isinstance(pred_output, np.ndarray) else pred_output

    assert result_gt.shape == pred_array.shape, \
        f'Shape mismatch! Expected {result_gt.shape}, got {pred_array.shape}'

    # Check values are close (NMF can have slight variations)
    assert np.allclose(result_gt, pred_array, rtol=1e-5), \
        f'Values mismatch!\nExpected:\n{result_gt}\nGot:\n{pred_array}'

    print("Output test passed!")


if __name__ == "__main__":
    # Test with ground truth input
    gt_input = {
        'self': {
            'n_components': 3,
            'init': None,
            'beta_loss': 'frobenius',
            'tol': 0.001,
            'max_iter': 200,
            'random_state': 0,
            'alpha_W': 0.0,
            'alpha_H': 'same',
            'l1_ratio': 0.0,
            'verbose': 0,
            'max_no_improvement': 10,
            'batch_size': 1024,
            'forget_factor': 0.7,
            'fresh_restarts': True,
            'fresh_restarts_max_iter': 30,
            'transform_max_iter': None,
            'n_features_in_': 5,
            '_n_components': 3,
            '_beta_loss': 2,
            '_batch_size': 6,
            '_rho': 0.7,
            '_gamma': 1.0,
            '_transform_max_iter': 200,
            '_components_numerator': '...',
            '_components_denominator': '...',
            '_ewa_cost': 0.06875329847666145,
            '_ewa_cost_min': 0.06877338125237134,
            '_no_improvement': 0,
            'reconstruction_err_': 0.9081689358928913,
            'n_components_': 3,
            'components_': '[[1.31688137, 0.03763225, 0.33914429, 1.95217318, 0.39567985], [0.16248088, 1.53022651, 0.70602284, 0.27344343, 0.04884479], [0.03521512, 0.47562   , 0.03721862, 0.81733954, 1.67062753]]',
            'n_iter_': 87,
            'n_steps_': 87
        },
        'args': {
            'X': '[[0.49671415, 0.1382643 , 0.64768854, 1.52302986, 0.23415337], [0.23413696, 1.57921282, 0.76743473, 0.46947439, 0.54256004], [0.46341769, 0.46572975, 0.24196227, 1.91328024, 1.72491783], [0.56228753, 1.01283112, 0.31424733, 0.90802408, 1.4123037 ], [1.46564877, 0.2257763 , 0.0675282 , 1.42474819, 0.54438272], [0.11092259, 1.15099358, 0.37569802, 0.60063869, 0.29169375]]'
        },
        'kwargs': {}
    }

    test_input(gt_input)

    # For output test, get the actual result
    X = np.array([
        [0.49671415, 0.1382643 , 0.64768854, 1.52302986, 0.23415337],
        [0.23413696, 1.57921282, 0.76743473, 0.46947439, 0.54256004],
        [0.46341769, 0.46572975, 0.24196227, 1.91328024, 1.72491783],
        [0.56228753, 1.01283112, 0.31424733, 0.90802408, 1.4123037 ],
        [1.46564877, 0.2257763 , 0.0675282 , 1.42474819, 0.54438272],
        [0.11092259, 1.15099358, 0.37569802, 0.60063869, 0.29169375]
    ])

    nmf = MiniBatchNMF(n_components=3, random_state=0)
    nmf.fit(X)
    gt_output = nmf.transform(X)

    print(f"Ground truth output type: {type(gt_output)}")
    print(f"Ground truth output shape: {gt_output.shape}")
    print(f"Ground truth output:\n{gt_output}")

    test_output(gt_output)
