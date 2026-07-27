# Problem: sympy@@sympy.core.expr.py@@__int___L352
# Module: sympy.core.expr
# Function: __int__
# Line: 352

from sympy import sympify


def test_input(pred_input):
    # Ground truth: sympify("1 + I") - a complex number
    obj_gt = sympify("1 + I")

    # Parse predicted input - self is a string representation
    obj_pred = sympify(pred_input['self'])

    # Both should raise TypeError when calling __int__ on complex numbers
    gt_error = None
    pred_error = None

    try:
        result_gt = int(obj_gt)
    except TypeError as e:
        gt_error = str(e)

    try:
        result_pred = int(obj_pred)
    except TypeError as e:
        pred_error = str(e)

    # If both raised errors, compare error messages
    if gt_error is not None and pred_error is not None:
        assert gt_error == pred_error, 'Prediction failed!'
    # If neither raised errors, compare results
    elif gt_error is None and pred_error is None:
        assert result_gt == result_pred, 'Prediction failed!'
    else:
        # One raised error, the other didn't
        assert False, 'Prediction failed! Error mismatch.'
