# Problem: sympy__sympy-12472@@sympy.simplify.radsimp.py@@handle_L750
# Module: sympy.simplify.radsimp
# Function: handle (nested function inside radsimp)
# Line: 750

from sympy import symbols, sympify
from sympy.simplify.radsimp import radsimp

# Define symbols
a, b = symbols('a b')


def test_input(pred_input):
    # Ground truth expression: 1/(a - b)
    expr_gt = 1/(a - b)

    # Parse the predicted expression from input
    expr_str = pred_input['args']['expr']
    expr_pred = sympify(expr_str)

    # Since handle is a nested function inside radsimp, we test through radsimp
    # radsimp calls handle internally
    result_gt = radsimp(expr_gt)
    result_pred = radsimp(expr_pred)

    assert result_gt == result_pred, f'Prediction failed! Expected: {result_gt}, Got: {result_pred}'


if __name__ == "__main__":
    # Test with the given input
    test_input_data = {
        "self": {},
        "args": {
            "expr": "1/(a - b)"
        },
        "kwargs": {}
    }

    test_input(test_input_data)
    print("Test passed!")
