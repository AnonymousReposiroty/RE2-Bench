# Problem: sympy__sympy-13146@@sympy.core.operations.py@@_eval_evalf_L296
# Module: sympy.core.operations
# Function: _eval_evalf
# Line: 296

from sympy import symbols, sqrt, pi, Add

# Create symbols for testing
x = symbols('x')


def test_input(pred_input):
    # Ground truth: Create an Add expression that has numeric and symbolic parts
    # _eval_evalf is called on AssocOp (Add/Mul) with a precision argument
    expr_gt = pi + sqrt(2) + x  # An Add expression with numbers and symbols

    # Ground truth precision
    prec_gt = 57

    # Predicted precision
    prec_pred = pred_input['args']['prec']

    # Call _eval_evalf on both with respective precision values
    result_gt = expr_gt._eval_evalf(prec=prec_gt)
    result_pred = expr_gt._eval_evalf(prec=prec_pred)

    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "prec": 57
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
