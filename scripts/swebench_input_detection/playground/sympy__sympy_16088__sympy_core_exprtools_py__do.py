# Problem: sympy__sympy-16088@@sympy.core.exprtools.py@@do_L1202
# Module: sympy.core.exprtools
# Function: do (nested function inside factor_terms)
# Line: 1202

from sympy import sympify, pi, sin
from sympy.core.exprtools import factor_terms


def test_input(pred_input):
    # Ground truth expression: sin(0.0644444444444444*pi)**2
    # do is a nested function inside factor_terms, so we test through factor_terms

    expr_gt = sin(0.0644444444444444*pi)**2

    # Parse predicted input
    expr_str = pred_input['args']['expr']
    expr_pred = sympify(expr_str)

    # Call factor_terms which internally calls do()
    result_gt = factor_terms(expr_gt)
    result_pred = factor_terms(expr_pred)

    # Compare results
    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "expr": "sin(0.0644444444444444*pi)**2"
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
