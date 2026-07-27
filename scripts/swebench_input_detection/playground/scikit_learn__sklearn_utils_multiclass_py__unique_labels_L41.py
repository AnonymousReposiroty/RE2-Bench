# Problem: scikit-learn@@sklearn.utils.multiclass.py@@unique_labels_L41
# Module: sklearn.utils.multiclass
# Function: unique_labels
# Line: 41

import numpy as np
from sklearn.utils.multiclass import unique_labels


def test_input(pred_input):
    # Ground truth: unique_labels([0, 1])
    # ys is a list of string representations, each element is a string like "[0, 1]"
    ys_gt = [[0, 1]]

    # Parse predicted ys from string representations
    ys_pred = []
    for y_str in pred_input['args']['ys']:
        # y_str is like "[0, 1]", parse it
        y_parsed = eval(y_str)
        ys_pred.append(y_parsed)

    # Call unique_labels with unpacked arguments
    result_gt = unique_labels(*ys_gt)
    result_pred = unique_labels(*ys_pred)

    # Compare results
    assert np.array_equal(result_gt, result_pred), 'Prediction failed!'
